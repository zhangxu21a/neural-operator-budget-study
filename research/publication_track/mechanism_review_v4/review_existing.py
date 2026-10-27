"""Read archived JSON only; no engine import, training, model selection or new evaluation."""
import os,json,csv,hashlib,statistics,collections
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/publication_mechanism_v4_mpl'
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]/"new_directions/python_deps"))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; O=Path(__file__).resolve().parent
LABEL_KEYS=['hf','coarse8','coarse32','phys8','phys32']
rows=[]; observations=[]; inputs={}; audit=[]
def read(f):
 b=f.read_bytes();inputs[str(f.relative_to(P))]=hashlib.sha256(b).hexdigest();return json.loads(b)
for study in ['formal_mg_v1','architecture_horizon_v2','matched_rate_v3']:
 for f in sorted((P/study/'formal').glob('*/result.json')):
  r=read(f);e=read(f.with_name('evaluation.json'));curve=read(f.with_name('curve.json'))
  assert not r['nonfinite_loss'] and r['initial_complete']
  for q in curve:
   observations.append(dict(study=study,run=f.parent.name,architecture=r.get('architecture','fno'),arm=r['arm'],lr=r['lr'],seed=r['seed'],**q,schedule_at_observation=r['lr']/(1+q['decision_ready']/60)))
  for cap,c in r['milestones'].items():
   cap=int(cap); ts=c['timings']; labels=sum(ts[k] for k in LABEL_KEYS); total=c['decision_ready']; residual=total-sum(ts.values())
   assert residual>=-1e-6 and total<=cap and c['selected_steps']<=c['steps_completed']
   # Replay only the checkpoint-selection bookkeeping from recorded assessments.
   available=[q for q in curve if q['decision_ready']<=total+1e-5]
   best=min(available,key=lambda q:q['validation'])
   assert best['steps']==c['selected_steps'] and abs(best['validation']-c['validation'])<1e-12
   v=e[str(cap)]
   for split in ['test','ood']:
    assert np.isfinite(v[split]['individual']).all()
    assert abs(np.mean(v[split]['individual'])-v[split]['relative_l2'])<1e-7
   row=dict(study=study,run=f.parent.name,architecture=r.get('architecture','fno'),arm=r['arm'],lr=r['lr'],seed=r['seed'],cap=cap,ready_seconds=total,label_seconds=labels,training_seconds=ts['training'],input_seconds=ts['input'],validation_seconds=ts['validation'],checkpoint_seconds=ts['checkpoint'],unclassified_seconds=residual,other_seconds=total-labels-ts['training'],completed_updates=c['steps_completed'],selected_updates=c['selected_steps'],updates_after_selected=c['steps_completed']-c['selected_steps'],selected_assessment_seconds=best['decision_ready'],selected_assessment_schedule=r['lr']/(1+best['decision_ready']/60),unique_labels_purchased=sum(c['counts'].values()),hf_labels=c['counts']['hf'],policy_labels=sum(c['counts'][k] for k in LABEL_KEYS if k!='hf'),id_pct=100*v['test']['relative_l2'],ood_pct=100*v['ood']['relative_l2'])
   assert np.isfinite(list(v for v in row.values() if isinstance(v,(int,float)))).all()
   rows.append(row)
for f in [P/'architecture_horizon_v2/study_engine.py',P/'formal_mg_v1/core.py']+[P/s/'protocol.json' for s in ['formal_mg_v1','architecture_horizon_v2','matched_rate_v3']]:inputs[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
def writecsv(name,rs):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
writecsv('endpoint_ledger.csv',rows);writecsv('assessment_observations.csv',observations)
groups=collections.defaultdict(list)
for r in rows:groups[(r['study'],r['architecture'],r['arm'],r['lr'],r['cap'])].append(r)
summaries=[]
metrics=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ['lr','seed','cap']]
for key,rs in groups.items():
 z=dict(zip(['study','architecture','arm','lr','cap'],key));z['n']=len(rs)
 for m in metrics:z[m+'_mean']=statistics.mean(r[m] for r in rs)
 z['selected_before_end_count']=sum(r['selected_updates']<r['completed_updates'] for r in rs)
 summaries.append(z)
writecsv('group_summary.csv',summaries)
(O/'summary.json').write_text(json.dumps(summaries,indent=2))
(O/'input_manifest.json').write_text(json.dumps({'sha256':inputs,'scope':'Archived formal JSON records, protocols and computation sources; no mutation or engine import.'},indent=2))
# Primary figure: latest balanced cohort only, all seeds and rates retained.
R=[r for r in rows if r['study']=='matched_rate_v3' and r['cap']==600]
keys=[(.002,'coarse32'),(.002,'phys32'),(.008,'coarse32'),(.008,'phys32')]
labels=['Coarse\n0.002','Corrected\n0.002','Coarse\n0.008','Corrected\n0.008']
colors={'coarse32':'#285F87','phys32':'#B87324'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(11.6,8.2),layout='constrained')
a=axs[0,0]
bottom=np.zeros(4)
for metric,label,col,hatch in [('label_seconds','Numerical labels','#B87324',''),('training_seconds','Training','#285F87',''),('other_seconds','Other charged time','#B9B9B9','//')]:
 vals=[np.mean([r[metric] for r in R if (r['lr'],r['arm'])==key]) for key in keys]
 a.bar(range(4),vals,bottom=bottom,color=col,label=label,hatch=hatch,width=.65)
 for i,(v,b) in enumerate(zip(vals,bottom)):
  if v>35:a.text(i,b+v/2,f'{v:.0f}',ha='center',va='center',color='white',fontsize=10)
 bottom+=vals
for i,key in enumerate(keys):
 rs=sorted([r for r in R if (r['lr'],r['arm'])==key],key=lambda r:r['seed'])
 a.scatter(np.array([i]*5)+np.linspace(-.12,.12,5),[r['ready_seconds'] for r in rs],c='black',s=10,zorder=4)
a.axhline(600,color='#444444',ls=':',lw=1);a.set_ylim(0,690);a.set_xticks(range(4),labels);a.set_ylabel('Charged wall time (s)');a.set_title('(a) Cost ledger at the 600 s cap',loc='left');a.legend(fontsize=8,ncol=3,loc='upper center',frameon=False)
a=axs[0,1]
for i,key in enumerate(keys):
 rs=sorted([r for r in R if (r['lr'],r['arm'])==key],key=lambda r:r['seed']);xs=i+np.linspace(-.15,.15,5)
 for x,r in zip(xs,rs):
  a.plot([x,x],[r['selected_updates'],r['completed_updates']],color='#999999',lw=1)
  a.scatter(x,r['completed_updates'],marker='o',color=colors[key[1]],s=22)
  a.scatter(x,r['selected_updates'],marker='s',facecolors='white',edgecolors=colors[key[1]],s=26,zorder=3)
a.scatter([],[],c='#444444',marker='o',label='Completed');a.scatter([],[],facecolors='white',edgecolors='#444444',marker='s',label='Selected checkpoint');a.legend(fontsize=8,frameon=False,loc='upper left');a.set_ylim(0,1700);a.set_xticks(range(4),labels);a.set_ylabel('Optimizer updates');a.set_title('(b) Spent updates vs retained model',loc='left');a.grid(axis='y',alpha=.15)
for ax,lr in zip(axs[1],[.002,.008]):
 for seed in sorted(set(r['seed'] for r in R)):
  pair=[next(r for r in R if r['seed']==seed and r['lr']==lr and r['arm']==arm) for arm in ['coarse32','phys32']]
  ax.plot([r['selected_updates'] for r in pair],[r['id_pct'] for r in pair],color='#B8B8B8',lw=.8,zorder=1)
 for arm,marker in [('coarse32','o'),('phys32','s')]:
  rs=[r for r in R if r['lr']==lr and r['arm']==arm];ax.scatter([r['selected_updates'] for r in rs],[r['id_pct'] for r in rs],c=colors[arm],marker=marker,s=42,label='Coarse' if arm=='coarse32' else 'Corrected',zorder=3)
 ax.set(xlim=(0,1600),ylim=(11,20),xlabel='Updates of validation-selected checkpoint',ylabel='ID relative L2 error (%)');ax.set_title(f'({"c" if lr==.002 else "d"}) Initial learning rate {lr}',loc='left');ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
fig.savefig(O/'cost_opportunity_error.pdf');fig.savefig(O/'cost_opportunity_error.png',dpi=180);plt.close(fig)
# Schedule diagnostic: function evaluated at recorded assessment completion, NOT actual step LR.
fig,axs=plt.subplots(1,2,figsize=(11.6,3.8),layout='constrained')
for ax,lr in zip(axs,[.002,.008]):
 for arm in ['coarse32','phys32']:
  for i,seed in enumerate(sorted(set(r['seed'] for r in R))):
   qs=[q for q in observations if q['study']=='matched_rate_v3' and q['lr']==lr and q['arm']==arm and q['seed']==seed and q['steps']>0]
   ax.plot([q['steps'] for q in qs],[q['schedule_at_observation']/lr for q in qs],color=colors[arm],alpha=.45,lw=1,ls='-' if arm=='coarse32' else '--',label=(('Coarse' if arm=='coarse32' else 'Corrected') if i==0 else None))
 ax.set(xlim=(0,1600),ylim=(0,1),xlabel='Completed updates at assessment',ylabel='Schedule value / initial learning rate');ax.set_title(f'Initial learning rate {lr}');ax.legend(frameon=False);ax.grid(alpha=.15)
fig.savefig(O/'schedule_diagnostic.pdf');fig.savefig(O/'schedule_diagnostic.png',dpi=180);plt.close(fig)
# Verify raw input immutability after the read-only analysis.
assert all(hashlib.sha256((P/f).read_bytes()).hexdigest()==h for f,h in inputs.items())
a={'runs':len(set((r['study'],r['run']) for r in rows)),'endpoints':len(rows),'cohort_endpoints':dict(collections.Counter(r['study'] for r in rows)),'all_budgets_and_ledgers_pass':True,'selection_bookkeeping_replayed_for_all_endpoints':True,'all_input_hashes_unchanged':True,'input_files':len(inputs),'no_training_or_new_test_evaluation':True,'posthoc_analysis':True,'exact_per_update_lr_available':False,'exact_sample_exposure_available':False}
(O/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
for s in summaries:
 if s['study']=='matched_rate_v3' and s['cap']==600:print(json.dumps(s))
# Preserved, explicitly post-hoc rate contrasts; never used to select a configuration.
import math
contrasts=[]
for arm in ['coarse32','phys32']:
 for metric in ['id_pct','ood_pct']:
  d={lr:{r['seed']:r[metric] for r in R if r['arm']==arm and r['lr']==lr} for lr in [.002,.008]}
  dif=[d[.008][seed]-d[.002][seed] for seed in sorted(d[.002])]
  mean=statistics.mean(dif);half=2.7764451051977987*statistics.stdev(dif)/math.sqrt(5)
  contrasts.append(dict(arm=arm,metric=metric,definition='0.008 minus 0.002, posthoc, 600 s',paired_differences_pp=dif,mean_pp=mean,exploratory_t95_pp=[mean-half,mean+half]))
(O/'within_policy_rate_contrasts.json').write_text(json.dumps(contrasts,indent=2))
