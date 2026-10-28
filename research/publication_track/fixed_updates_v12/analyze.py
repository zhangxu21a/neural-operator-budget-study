"""Frozen descriptive analysis; no tuning, reruns, exclusions or editorial mutations."""
import pathlib,json,math,statistics,hashlib
R=pathlib.Path(__file__).resolve().parent
p=json.loads((R/'protocol.json').read_text());rows=[];audits=[];failures=[]
for seed,rate,arm in p['jobs']:
 d=R/'formal'/f'{arm}_lr{rate}_s{seed}';r=json.loads((d/'result.json').read_text());e=json.loads((d/'evaluation.json').read_text());data=R/'data'/str(seed);m=json.loads((data/'manifest.json').read_text())
 assert r['initial_state_hash']==m['initial_state_hash'];assert r['data_manifest_sha256']==hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
 updates=[json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()]
 for i,u in enumerate(updates):assert u['step']==i+1 and abs(u['lr']-rate/(1+i/p['lr_tau_updates']))<1e-15
 if r['failure']:failures.append({'seed':seed,'rate':rate,'arm':arm,'failure':r['failure']})
 else:assert len(updates)==1400 and set(r['endpoints'])=={'800','1400'}
 for n in p['endpoints']:
  if str(n) not in e:continue
  assert r['endpoints'][str(n)]['selected_step']<=n
  for which in ['last','selected']:
   for split,N in [('test',128),('ood',64)]:
    vals=e[str(n)][which][split];assert len(vals['individual'])==N
    finite=all(math.isfinite(x) for x in vals['individual'])
    if finite:assert abs(statistics.fmean(vals['individual'])-vals['relative_l2'])<1e-10
    rows.append({'seed':seed,'rate':rate,'arm':arm,'updates':n,'model':which,'split':split,'error_pct':100*vals['relative_l2'] if finite else None,'finite':finite,'selected_step':r['endpoints'][str(n)]['selected_step'],'run_wall':r['wall_seconds']})
 audits.append({'run':d.name,'updates_logged':len(updates),'failure':r['failure'],'initial_hash':r['initial_state_hash']})
# Exact sequence equality across all four policies/rates for each seed, including all successful prefixes.
for seed in p['seeds']:
 sequences=[]
 for rate in p['rates']:
  for arm in p['arms']:
   f=R/'formal'/f'{arm}_lr{rate}_s{seed}'/'updates.jsonl';sequences.append([json.loads(x)['batch_rows'] for x in f.read_text().splitlines()])
 n=min(map(len,sequences));assert all(x[:n]==sequences[0][:n] for x in sequences)

def estimate(ds):
 valid=[x for x in ds if x is not None];n=len(valid)
 if n!=5:return {'n':n,'planned_n':5,'status':'incomplete_or_nonfinite','all_differences_pp':ds,'mean_pp':None,'ci95_pp':None}
 mean=statistics.fmean(valid);se=statistics.stdev(valid)/math.sqrt(n);pos=sum(x>0 for x in valid);neg=sum(x<0 for x in valid);nonzero=pos+neg
 signp=min(1.,2*sum(math.comb(nonzero,k) for k in range(min(pos,neg)+1))/2**nonzero) if nonzero else 1.
 return {'n':5,'all_differences_pp':valid,'mean_pp':mean,'sd_pp':statistics.stdev(valid),'ci95_pp':[mean-2.7764451051977987*se,mean+2.7764451051977987*se],'negative':neg,'positive':pos,'ties':5-nonzero,'two_sided_sign_p':signp,'inference':'exploratory paired t interval; no multiplicity adjustment; no p<.05 claim from 5 identical signs'}
lookup={(r['seed'],r['rate'],r['arm'],r['updates'],r['model'],r['split']):r['error_pct'] for r in rows}
def value(seed,rate,arm,n,model,split):return lookup.get((seed,rate,arm,n,model,split))
def contrast(terms):
 ds=[]
 for seed in p['seeds']:
  vals=[(coef,value(seed,*args)) for coef,args in terms]
  ds.append(sum(coef*x for coef,x in vals) if all(x is not None for _,x in vals) else None)
 return estimate(ds)
groups=[];contrasts=[]
for model in ['last','selected']:
 for split in ['test','ood']:
  for rate in p['rates']:
   for n in p['endpoints']:
    for arm in p['arms']:
     vals=[value(seed,rate,arm,n,model,split) for seed in p['seeds']]
     groups.append({'model':model,'split':split,'rate':rate,'updates':n,'arm':arm,'values_pct':vals,'mean_pct':statistics.fmean(vals) if all(x is not None for x in vals) else None})
    contrasts.append({'type':'corrected_minus_coarse','model':model,'split':split,'rate':rate,'updates':n,**contrast([(1,(rate,'phys32',n,model,split)),(-1,(rate,'coarse32',n,model,split))])})
   for arm in p['arms']:
    contrasts.append({'type':'1400_minus_800','model':model,'split':split,'rate':rate,'arm':arm,**contrast([(1,(rate,arm,1400,model,split)),(-1,(rate,arm,800,model,split))])})
   contrasts.append({'type':'target_by_updates_interaction','model':model,'split':split,'rate':rate,**contrast([(1,(rate,'phys32',1400,model,split)),(-1,(rate,'coarse32',1400,model,split)),(-1,(rate,'phys32',800,model,split)),(1,(rate,'coarse32',800,model,split))])})
  for n in p['endpoints']:
   contrasts.append({'type':'target_by_rate_interaction','model':model,'split':split,'updates':n,**contrast([(1,(.008,'phys32',n,model,split)),(-1,(.008,'coarse32',n,model,split)),(-1,(.002,'phys32',n,model,split)),(1,(.002,'coarse32',n,model,split))])})
(R/'summary.json').write_text(json.dumps({'groups':groups,'contrasts':contrasts,'failures':failures,'primary':'last model, ID, target contrast within both rates and both update endpoints','secondary':'selected model, OOD, step and interaction contrasts; correlated outcomes, not independent replicates'},indent=2))
(R/'all_endpoints.json').write_text(json.dumps(rows,indent=2));(R/'audit.json').write_text(json.dumps({'runs':audits,'matching_batch_sequences':True,'matching_initial_models':True,'sources_match':all(hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h for f,h in p['sources'].items()),'failures':failures,'all_nonfinite_retained':True},indent=2))
lines=['# Fixed-update diagnostic: frozen analysis','', 'Primary results use last iterates; selected-checkpoint results remain separate. All rates and seeds retained. No multiplicity-adjusted significance claims.','', '| Rate | Updates | Coarse ID (%) | Corrected ID (%) | Corrected - coarse (pp) | Paired t95 interval |','|---|---|---|---|---|---|']
for rate in p['rates']:
 for n in p['endpoints']:
  gs=[g for g in groups if g['model']=='last' and g['split']=='test' and g['rate']==rate and g['updates']==n];c=next(c for c in contrasts if c['type']=='corrected_minus_coarse' and c['model']=='last' and c['split']=='test' and c['rate']==rate and c['updates']==n)
  lines.append(f"| {rate} | {n} | {gs[0]['mean_pct']} | {gs[1]['mean_pct']} | {c['mean_pp']} | {c['ci95_pp']} |")
lines+=['','All groups, seed differences, OOD/selected endpoints and interactions are in summary.json.','Failures: '+str(failures),'','Interpretation must compare new within-study controlled contrasts, not absolute errors across independent reference sets. This does not estimate mediation fractions for the old wall-time study or isolate sample-exposure count from update count.']
(R/'results_report.md').write_text('\n'.join(lines)+'\n');print('Analysis complete; failures retained:',len(failures),flush=True)
