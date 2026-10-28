from pathlib import Path
import shutil,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent.parent; O=Path(__file__).resolve().parent
# Read-only sensitivity: average across paired seeds first, retain shared test-input dimension.
records=[]; hashes={}
def read(p):
 b=p.read_bytes();hashes[str(p.relative_to(R))]=hashlib.sha256(b).hexdigest();return json.loads(b)
def compare(cohort,a,b,cap,label):
 da={p.parent.name.rsplit('_s',1)[1]:p for p in (R/cohort/'formal').glob(a+'_s*/evaluation.json')}
 db={p.parent.name.rsplit('_s',1)[1]:p for p in (R/cohort/'formal').glob(b+'_s*/evaluation.json')}
 assert len(da)==len(db)==5 and da.keys()==db.keys()
 for split in ['test','ood']:
  aa=np.array([read(da[s])[str(cap)][split]['individual'] for s in sorted(da)])*100
  bb=np.array([read(db[s])[str(cap)][split]['individual'] for s in sorted(db)])*100
  assert aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
  d=(aa-bb).mean(axis=0); n=len(d); k=int(np.floor(.1*n))
  trimmed=np.sort(d)[k:n-k].mean();loo=(d.sum()-d)/(n-1)
  # symmetric difficulty exclusion, not outcome-sign selection
  keep=np.argsort(((aa+bb)/2).mean(axis=0))[:n-int(np.ceil(.1*n))]
  records.append(dict(cohort=cohort,comparison=label,cap=cap,split=split,n=n,mean=float(d.mean()),median=float(np.median(d)),trimmed10=float(trimmed),drop_hardest10=float(d[keep].mean()),loo_min=float(loo.min()),loo_max=float(loo.max())))
compare('formal_mg_v1','hf128','hf_online',180,'FNO baseline: reuse - online')
for arch,cap in [('fno',180),('fno',600),('deeponet',180)]: compare('architecture_horizon_v2',arch+'_hf128',arch+'_hf_online',cap,arch+' horizon: reuse - online')
for rate in ['0.002','0.008']: compare('matched_rate_v3','phys32_lr'+rate,'coarse32_lr'+rate,600,'Corrected - coarse, '+rate)
(O/'robustness.json').write_text(json.dumps(records,indent=2))
(O/'analysis_input_hashes.json').write_text(json.dumps(hashes,indent=2))
print("Verified saved errors; wrote robustness.json and analysis_input_hashes.json")
