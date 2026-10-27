import pathlib,json,hashlib,csv,statistics,collections
R=pathlib.Path(__file__).resolve().parents[1];M=R.parent/'mainline_revision'/'darcy_mechanism'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((M/'protocol.json').read_text())
checks={p:hashfile(M/p)==v for p,v in protocol['source_hashes'].items()}
files=[p for p in M.rglob('*') if p.is_file() and not p.name.startswith('._') and '__pycache__' not in p.parts and 'mpl_config' not in p.parts]
manifest={str(p.relative_to(R.parent)): {'bytes':p.stat().st_size,'sha256':hashfile(p)} for p in files}
(R/'evidence'/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
rows=list(csv.DictReader((M/'all_runs.csv').open()));keys=[tuple(r[k] for k in ['method','lr','seed','budget']) for r in rows]
assert len(keys)==len(set(keys))==60
selected=json.loads((M/'selected.json').read_text())
means=[]
for (m,lr,b),rr in __import__('itertools').groupby(sorted(rows,key=lambda r:(r['method'],r['lr'],r['budget'])),key=lambda r:(r['method'],r['lr'],r['budget'])):
 rr=list(rr);assert len(rr)==3
 d={'method':m,'lr':float(lr),'budget':int(b),'n':3}
 for metric in ['test','ood','fine','steps','scoring_seconds']:
  a=[float(r[metric]) for r in rr];d[metric+'_mean']=statistics.mean(a);d[metric+'_sd']=statistics.stdev(a)
 means.append(d)
pairs=[]
for budget in ['300','1200']:
 for a,b in [('phys_only','hf'),('data','phys_only'),('gradient','uniform')]:
  aa={r['seed']:r for r in rows if r['method']==a and r['lr']=='0.008' and r['budget']==budget};bb={r['seed']:r for r in rows if r['method']==b and r['lr']=='0.008' and r['budget']==budget}
  for metric in ['test','ood']:
   dif=[100*(float(aa[s][metric])-float(bb[s][metric])) for s in sorted(aa)]
   pairs.append({'left':a,'right':b,'budget':int(budget),'metric':metric,'per_seed_difference_percentage_points':dif,'mean_difference_pp':statistics.mean(dif),'sd_difference_pp':statistics.stdev(dif),'wins':sum(x<0 for x in dif),'n':len(dif),'two_sided_sign_test_min_possible_p':.25})
formal=list(M.glob('formal/*/result.json'));tuning=list(M.glob('tuning/*/result.json'))
audit={'frozen_source_checks':checks,'all_frozen_sources_match':all(checks.values()),'formal_trajectories':len(formal),'tuning_runs':len(tuning),'budget_points':len(rows),'unique_budget_points':len(set(keys)),'manifest_files':len(manifest),'means':means,'paired':pairs,'notes':['One trajectory has two dependent budget points.','n=3 independent optimization seeds, shared train/test inputs; not three independent datasets.','No population-level claim or fitted scaling law.']}
assert len(formal)==30 and len(tuning)==48 and all(checks.values())
(R/'evidence'/'independent_audit.json').write_text(json.dumps(audit,indent=2))
with (R/'evidence'/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=means[0].keys());w.writeheader();w.writerows(means)
print(json.dumps({k:v for k,v in audit.items() if k not in ['means','paired','frozen_source_checks']},indent=2))
