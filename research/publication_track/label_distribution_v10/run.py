import os,sys,pathlib,json,hashlib,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
O=pathlib.Path(__file__).resolve().parent;R=O.parent
sys.path.insert(0,str(R/'formal_mg_v1'))
import core
import numpy as np
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'protocol.json').exists(),'Use a new directory; do not overwrite protocol or outcomes'
sources=json.loads((R/'matched_rate_v3/protocol.json').read_text())['sources']
assert all(h(pathlib.Path(p))==v for p,v in sources.items())
params=core.parameters(2026102801,512,'channel');np.save(O/'parameters.npy',params)
# Compare only input parameters, never saved test errors or predictions.
for c in ['formal_mg_v1','architecture_horizon_v2','matched_rate_v3']:
 with np.load(R/c/'reference.npz') as ref:
  for split in ['validation','test','ood']:
   old={x.tobytes() for x in ref[split+'_p']};assert not any(x.tobytes() in old for x in params)
protocol={'n':512,'seed':2026102801,'distribution':'independent channel inputs, gain 2','nc':32,'nf':512,'observation_grid':129,'hf':'frozen MGCG rtol1e-8','inputs_sha256':h(O/'parameters.npy'),'script_sha256':h(pathlib.Path(__file__)),'sources':sources,'selection':'All frozen 512 inputs; no filtering, no tuning, no training; no outcomes inspected before protocol','cost':'All new HF/coarse/corrected solves separately charged as post-hoc diagnostic expense, outside original budgets','interval':'paired mean difference percentile bootstrap; 20000 resamples of input triples, RNG2026102802; descriptive 95% interval conditional on sampled distribution','quantiles':[.5,.9,.95],'no_reference_input_overlap':True}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
oracle=core.Oracle(params);rows=[];ys={k:[] for k in ['hf','coarse32','phys32']};start=time.perf_counter()
for i in range(512):
 row={'input_index':i}
 for kind in ys:
  t=time.perf_counter();z=oracle.get(kind,i);row[kind+'_seconds']=time.perf_counter()-t;ys[kind].append(z)
 hf=ys['hf'][-1].astype('float64');den=np.linalg.norm(hf);assert den>0
 for k in ['coarse32','phys32']:row[k+'_error']=float(np.linalg.norm(ys[k][-1].astype('float64')-hf)/den)
 row['delta']=row['phys32_error']-row['coarse32_error'];assert np.isfinite(list(row.values())).all();rows.append(row)
 (O/'status.json').write_text(json.dumps({'done':i+1,'total':512,'elapsed':time.perf_counter()-start}))
 if (i+1)%64==0:print('Completed',i+1,'/512',flush=True)
np.savez_compressed(O/'paired_labels.npz',params=params,**{k:np.stack(v) for k,v in ys.items()})
(O/'individual.json').write_text(json.dumps(rows,indent=2))
assert all(h(pathlib.Path(p))==v for p,v in sources.items())
(O/'completion.json').write_text(json.dumps({'complete':True,'n':len(rows),'new_solves':1536,'seconds':time.perf_counter()-start,'solver_seconds':oracle.times,'max_hf_residual':max(oracle.residuals),'max_hf_iterations':max(oracle.iterations),'frozen_sources_unchanged':True},indent=2))
