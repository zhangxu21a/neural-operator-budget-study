"""Complete pre-existing validation splits; no test/OOD reads, training or selection."""
import os,sys,pathlib,json,time,hashlib
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.dont_write_bytecode=True
OUT=pathlib.Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'formal_mg_v1'))
import core
import numpy as np
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cohorts=['formal_mg_v1','architecture_horizon_v2','matched_rate_v3']
sources=json.loads((ROOT/'matched_rate_v3/protocol.json').read_text())['sources']
for name,expected in sources.items():assert hashfile(pathlib.Path(name))==expected,name
refs={c:hashfile(ROOT/c/'reference.npz') for c in cohorts}
protocol={'cohorts':cohorts,'split':'validation','inputs_per_cohort':32,'total_pairs':96,'gain':2,'coarse_intervals':32,'fine_intervals':512,'observation_grid':129,'target_policies':['coarse32','phys32'],'hf':'existing scaled float32 MGCG target, rtol1e-8','metric':'Euclidean norm error on all 129x129 nodes, denominator existing HF target norm; scaling .05 cancels','selection':'All existing development inputs, no filtering, no tests or OOD, no tuning; separate cohorts','no_new_training':True,'sources':sources,'references':refs,'script_sha256':hashfile(pathlib.Path(__file__)),'timing':'Diagnostic label-generation wall time only, not controlled original performance timing'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];groups={};start=time.perf_counter()
for cohort in cohorts:
 with np.load(ROOT/cohort/'reference.npz') as data:
  params=data['validation_p'].copy();hf=data['validation_y'].copy().reshape(32,-1)
 assert len(params)==32 and hf.shape==(32,129*129)
 oracle=core.Oracle(params);targets={k:[] for k in ['coarse32','phys32']}
 for i in range(32):
  row={'cohort':cohort,'input_index':i}
  for kind in targets:
   t=time.perf_counter();z=oracle.get(kind,i);dt=time.perf_counter()-t
   assert np.isfinite(z).all() and np.linalg.norm(hf[i])>0
   targets[kind].append(z);row[kind+'_relative_l2']=float(np.linalg.norm(z.astype('float64')-hf[i])/np.linalg.norm(hf[i].astype('float64')));row[kind+'_seconds']=dt
  row['corrected_minus_coarse']=row['phys32_relative_l2']-row['coarse32_relative_l2'];rows.append(row)
  (OUT/'status.json').write_text(json.dumps({'done':len(rows),'total':96,'cohort':cohort,'elapsed':time.perf_counter()-start}))
 np.savez_compressed(OUT/(cohort+'_paired_labels.npz'),params=params,hf=hf,coarse32=np.stack(targets['coarse32']),phys32=np.stack(targets['phys32']))
 group={}
 for kind in targets:
  x=np.array([r[kind+'_relative_l2'] for r in rows if r['cohort']==cohort])*100
  group[kind]={k:float(v) for k,v in zip(['min','q25','median','q75','p95','max','mean'],[*np.quantile(x,[0,.25,.5,.75,.95,1]),x.mean()])}
 diff=np.array([r['corrected_minus_coarse'] for r in rows if r['cohort']==cohort])*100
 group['paired']={'mean_difference_pp':float(diff.mean()),'median_difference_pp':float(np.median(diff)),'improved':int((diff<0).sum()),'worse':int((diff>0).sum()),'ties':int((diff==0).sum()),'n':32}
 groups[cohort]=group
(OUT/'individual.json').write_text(json.dumps(rows,indent=2));(OUT/'summary.json').write_text(json.dumps(groups,indent=2))
assert all(hashfile(pathlib.Path(n))==h for n,h in sources.items())
assert all(hashfile(ROOT/c/'reference.npz')==h for c,h in refs.items())
(OUT/'completion.json').write_text(json.dumps({'complete':True,'pairs':len(rows),'new_label_targets':len(rows)*2,'new_training':False,'test_or_ood_used':False,'sources_and_references_unchanged':True,'elapsed_seconds':time.perf_counter()-start},indent=2))
print(json.dumps(groups,indent=2),flush=True)
