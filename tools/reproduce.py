"""Portable, standard-library-only reanalysis. Never imports or runs training code."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--verify-assets',action='store_true');a=ap.parse_args();root=a.root.resolve();out=a.out.resolve()
if out.exists():raise SystemExit('Refusing existing output directory; choose a fresh path.')
manifest=json.loads((root/'MANIFEST.json').read_text());verified=0
for row in manifest['files']:
 if row['tier']=='assets' and not a.verify_assets:continue
 f=root/row['path']
 h=hashlib.sha256()
 with f.open('rb') as z:
  for b in iter(lambda:z.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=row['sha256']:raise RuntimeError('Hash mismatch: '+row['path'])
 verified+=1
P=root/'research/publication_track';rows=[];costs=[]
expected={'formal_mg_v1':(35,70,21),'architecture_horizon_v2':(45,130,27),'matched_rate_v3':(20,80,0)}
for study,(nrun,nend,ntune) in expected.items():
 ds=sorted((P/study/'formal').glob('*/result.json'));assert len(ds)==nrun;before=len(rows)
 tunes=list((P/study/'tuning').glob('*/result.json'));assert len(tunes)==ntune
 for f in ds:
  r=json.loads(f.read_text());e=json.loads(f.with_name('evaluation.json').read_text());curve=json.loads(f.with_name('curve.json').read_text())
  assert not r['nonfinite_loss'] and r['initial_complete']
  for cap,c in r['milestones'].items():
   assert c['decision_ready']<=int(cap) and c['selected_steps']<=c['steps_completed']
   label=sum(c['timings'][k] for k in ['hf','coarse8','coarse32','phys8','phys32'])
   assert c['decision_ready']-sum(c['timings'].values())>=-1e-6
   available=[q for q in curve if q['decision_ready']<=c['decision_ready']+1e-5];best=min(available,key=lambda q:q['validation'])
   assert best['steps']==c['selected_steps'] and abs(best['validation']-c['validation'])<1e-12
   vals={}
   for split in ['test','ood']:
    x=e[cap][split]['individual'];assert all(math.isfinite(v) for v in x)
    vals[split]=statistics.fmean(x)*100
    assert abs(vals[split]/100-e[cap][split]['relative_l2'])<1e-7
   rows.append(dict(study=study,architecture=r.get('architecture','fno'),arm=r['arm'],lr=r['lr'],seed=r['seed'],cap=int(cap),id_pct=vals['test'],ood_pct=vals['ood'],label_seconds=label,train_seconds=c['timings']['training'],ready_seconds=c['decision_ready'],completed_updates=c['steps_completed'],selected_updates=c['selected_steps']))
  costs.append(dict(study=study,run=f.parent.name,wall=r['wall'],export_seconds=r.get('export_seconds',None),offline_evaluation_seconds=e.get('offline_seconds',None)))
 assert len(rows)-before==nend
out.mkdir(parents=True)
def csvout(name,rs):
 with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
csvout('endpoints.csv',rows);csvout('excluded_costs_per_run.csv',costs)
groups={}
for r in rows:groups.setdefault((r['study'],r['architecture'],r['arm'],r['lr'],r['cap']),[]).append(r)
means=[]
for k,rs in sorted(groups.items()):
 z=dict(zip(['study','architecture','arm','lr','cap'],k));z['n']=len(rs)
 for m in ['id_pct','ood_pct','label_seconds','train_seconds','ready_seconds','completed_updates','selected_updates']:z[m+'_mean']=statistics.fmean(r[m] for r in rs);z[m+'_sd']=statistics.stdev(r[m] for r in rs)
 means.append(z)
csvout('group_summary.csv',means)
contrasts=[]
for cap in [120,180,360,600]:
 for lr in [.002,.008]:
  for metric in ['id_pct','ood_pct']:
   rr=[r for r in rows if r['study']=='matched_rate_v3' and r['cap']==cap and r['lr']==lr]
   byarm={arm:{r['seed']:r[metric] for r in rr if r['arm']==arm} for arm in ['phys32','coarse32']}
   ds=[byarm['phys32'][s]-byarm['coarse32'][s] for s in sorted(byarm['phys32'])];mu=statistics.fmean(ds);half=2.7764451051977987*statistics.stdev(ds)/math.sqrt(5)
   contrasts.append(dict(cap=cap,lr=lr,metric=metric,definition='corrected minus coarse',differences_pp=ds,mean_pp=mu,exploratory_t95_pp=[mu-half,mu+half]))
(out/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2))
# Verify the independently regenerated group means against the released prior review.
prior=json.loads((P/'mechanism_review_v4/summary.json').read_text())
for x in prior:
 y=next(z for z in means if all(z[k]==x[k] for k in ['study','architecture','arm','lr','cap']))
 for k in ['id_pct_mean','ood_pct_mean','label_seconds_mean','completed_updates_mean','selected_updates_mean']:assert abs(x[k]-y[k])<1e-5,(k,x[k],y[k])
report={'verified_files':verified,'formal_runs':sum(v[0] for v in expected.values()),'tuning_runs':sum(v[2] for v in expected.values()),'dependent_endpoints':len(rows),'all_source_checks_pass':True,'all_budgets_and_selection_checks_pass':True,'released_means_reproduced':True,'new_training':False,'new_model_evaluation':False,'assets_checked':a.verify_assets,'scope':'Three prospective cohorts. Legacy appendix remains a separate archived cohort; not pooled.'}
(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
