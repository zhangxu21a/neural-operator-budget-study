"""Read-only portable audit of v9/v10/v12 records. No training or model evaluation."""
from pathlib import Path
import argparse,json,hashlib,math,statistics
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--verify-assets',action='store_true');a=p.parse_args();R=Path(__file__).resolve().parents[1];P=R/'research/publication_track'
assert not a.out.exists(),'Choose a new output directory'
def sha(f):
 d=hashlib.sha256()
 with f.open('rb') as z:
  while b:=z.read(8*1024*1024):d.update(b)
 return d.hexdigest()
m=json.loads((R/'V12_MANIFEST.json').read_text());checked=0
for row in m['files']:
 if row['tier']=='assets' and not a.verify_assets:continue
 assert sha(R/row['path'])==row['sha256'],row['path'];checked+=1
for study in ['development_label_v9','label_distribution_v10','fixed_updates_v12']:
 proto=json.loads((P/study/'protocol.json').read_text())
 for f,h in proto['sources'].items():assert sha(R/m['frozen_source_map'][f])==h,f
D=P/'fixed_updates_v12';proto=json.loads((D/'protocol.json').read_text());assert sha(D/'protocol.json')==(D/'protocol.sha256').read_text().strip()
values={};sequences={};initial={};count=0
for seed,rate,arm in proto['jobs']:
 d=D/'formal'/f'{arm}_lr{rate}_s{seed}';r=json.loads((d/'result.json').read_text());e=json.loads((d/'evaluation.json').read_text());dm=json.loads((D/'data'/str(seed)/'manifest.json').read_text());assert sha(D/'data'/str(seed)/'manifest.json')==r['data_manifest_sha256'];assert r['initial_state_hash']==dm['initial_state_hash'];assert not r['failure']
 log=[json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()];assert len(log)==1400
 seq=[x['batch_rows'] for x in log]
 if seed in sequences:assert sequences[seed]==seq and initial[seed]==r['initial_state_hash']
 sequences[seed]=seq;initial[seed]=r['initial_state_hash']
 for i,u in enumerate(log):assert u['step']==i+1 and abs(u['lr']-rate/(1+i/160))<1e-15
 for n in [800,1400]:
  for model in ['last','selected']:
   for split,N in [('test',128),('ood',64)]:
    v=e[str(n)][model][split];xs=v['individual'];assert len(xs)==N and all(math.isfinite(x) for x in xs);mu=statistics.fmean(xs);assert abs(mu-v['relative_l2'])<1e-10;values[seed,rate,arm,n,model,split]=mu*100;count+=1
summary=json.loads((D/'summary.json').read_text());verified_contrasts=[]
for c in summary['contrasts']:
 def v(seed,rate,arm,n):return values[seed,rate,arm,n,c['model'],c['split']]
 ds=[]
 for seed in proto['seeds']:
  if c['type']=='corrected_minus_coarse':x=v(seed,c['rate'],'phys32',c['updates'])-v(seed,c['rate'],'coarse32',c['updates'])
  elif c['type']=='1400_minus_800':x=v(seed,c['rate'],c['arm'],1400)-v(seed,c['rate'],c['arm'],800)
  elif c['type']=='target_by_updates_interaction':
   x=v(seed,c['rate'],'phys32',1400)-v(seed,c['rate'],'coarse32',1400)-v(seed,c['rate'],'phys32',800)+v(seed,c['rate'],'coarse32',800)
  elif c['type']=='target_by_rate_interaction':
   x=v(seed,.008,'phys32',c['updates'])-v(seed,.008,'coarse32',c['updates'])-v(seed,.002,'phys32',c['updates'])+v(seed,.002,'coarse32',c['updates'])
  else:raise AssertionError(c['type'])
  ds.append(x)
 assert all(abs(x-y)<1e-10 for x,y in zip(ds,c['all_differences_pp']));assert abs(statistics.fmean(ds)-c['mean_pp'])<1e-10
 half=2.7764451051977987*statistics.stdev(ds)/math.sqrt(5);mu=statistics.fmean(ds)
 assert abs(c['ci95_pp'][0]-(mu-half))<1e-10 and abs(c['ci95_pp'][1]-(mu+half))<1e-10
 verified_contrasts.append(c)
rows=json.loads((P/'label_distribution_v10/individual.json').read_text());s=json.loads((P/'label_distribution_v10/summary.json').read_text());assert len(rows)==512
for key,field in [('coarse','coarse32_error'),('corrected','phys32_error'),('delta_pp','delta')]:assert abs(statistics.fmean(x[field]*100 for x in rows)-s[key]['mean'])<1e-10
assert sum(x['delta']<0 for x in rows)==s['improved']
drows=json.loads((P/'development_label_v9/individual.json').read_text());dsummary=json.loads((P/'development_label_v9/summary.json').read_text());assert len(drows)==96
for cohort in ['formal_mg_v1','architecture_horizon_v2','matched_rate_v3']:
 rr=[x for x in drows if x['cohort']==cohort];assert len(rr)==32
 for key in ['coarse32','phys32']:assert abs(statistics.fmean(x[key+'_relative_l2']*100 for x in rr)-dsummary[cohort][key]['mean'])<1e-10
a.out.mkdir(parents=True);result={'files_checked':checked,'frozen_sources_verified':len(m['frozen_source_map']),'formal_trajectories':20,'correlated_endpoints':40,'evaluation_records':count,'matched_initialization_and_batches':True,'per_step_rates_verified':True,'all_paired_contrasts_verified':len(verified_contrasts),'fresh_label_inputs':512,'assets_checked':a.verify_assets,'new_training':False,'new_predictions':False,'scope':'hashes and saved records; no claim of full cross-machine training replay'};(a.out/'verification.json').write_text(json.dumps(result,indent=2));(a.out/'paired_contrasts.json').write_text(json.dumps(verified_contrasts,indent=2));print(json.dumps(result,indent=2))
