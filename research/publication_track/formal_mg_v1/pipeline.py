import core
from core import *
import fcntl,subprocess,traceback,random
lock=(ROOT/'.timing.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(name,x):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def event(stage,**kw):
 d={'stage':stage,'time_unix':time.time(),'pid':os.getpid(),**kw};write('status.json',d);print(json.dumps(d),flush=True)
def sources():
 files=[ROOT/'core.py',ROOT/'pipeline.py',ROOT/'analyze.py']
 base=json.loads((RESEARCH/'mainline_revision/darcy_mechanism/protocol.json').read_text())
 files +=[(RESEARCH/'mainline_revision/darcy_mechanism'/p).resolve() for p in base['source_hashes']]
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def wait_idle():
 while True:
  other=[]
  for line in subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True).splitlines()[1:]:
   try:pid,cpu,comm=line.strip().split(None,2)
   except ValueError:continue
   if int(pid)!=os.getpid() and 'python' in comm.lower() and float(cpu)>30:other.append({'pid':int(pid),'cpu_percent':float(cpu)})
  if not other:return
  event('waiting_for_other_compute',processes=other);time.sleep(30)
def reference():
 path=ROOT/'reference.npz'
 if path.exists():return dict(np.load(path))
 ts=time.perf_counter();out={}
 for split,seed,count in [('validation',62001,32),('test',62002,128),('ood',62003,64)]:
  p=parameters(seed,count,'channel')
  if split=='ood':p[:,:16]*=1.25
  oracle=Oracle(p);out[split+'_p']=p;out[split+'_x']=np.stack([make_input(q) for q in p])[:,None];out[split+'_y']=np.stack([oracle.get('hf',i) for i in range(count)])
  event('reference',split=split,count=count,max_residual=max(oracle.residuals))
 np.savez_compressed(path,**out);write('reference_manifest.json',{'wall_seconds':time.perf_counter()-ts,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seeds':[62001,62002,62003],'counts':[32,128,64],'excluded_from_run_budget':True});return out

def validate(ref):
 if (ROOT/'validation.json').exists():return
 event('validation')
 from solver import validate as manufactured
 checks={'manufactured':manufactured(),'solver':[]}
 p=parameters(63001,3,'channel');oracle=Oracle(p)
 for i,q in enumerate(p):
  A=matrix(coefficient(q,512)**2);direct=field(spsolve(A,np.ones(A.shape[0])),512)[::4,::4].ravel()/SCALE;fast=oracle.get('hf',i)
  err=float(np.linalg.norm(fast-direct)/np.linalg.norm(direct));assert err<1e-6
  checks['solver'].append({'relative_error_to_direct':err,'residual':oracle.residuals[-1]})
 for arm,budget in [('coarse8',20),('phys32',20),('hf128',45)]:
  r=run(arm,71,.008,budget,ROOT/'smoke'/arm,ref,phase='smoke',pool=256)
  assert r['wall']<=budget and not r['nonfinite_loss'];checks[arm]={'wall':r['wall'],'steps':r['steps'],'initial_complete':r['initial_complete']}
 checks['passed']=True;write('validation.json',checks)

def main():
 h=sources()
 protocol={'arms':ARMS,'formal_seeds':[101,202,303,404,505],'formal_train_parameter_seeds':[71101,71202,71303,71404,71505],'tuning_seed':17,'learning_rates':[.002,.008,.032],'tuning_seconds':60,'tuning_budget_per_arm':180,'formal_seconds':180,'formal_checkpoints':[120,180],'pool':8192,'gain':2,'nf':512,'output_grid_intervals':128,'family':'channel','hf_solver':'Galerkin geometric MG preconditioned CG, rtol1e-8; shared geometry interpolation built within each run; coefficient-dependent hierarchy rebuilt for every new label','coarse_grids':[8,32],'hf_common_initial':32,'hf_reuse_sizes':[32,128],'factor_scope':'Hold gain and fine grid fixed; isolate coarse-space size. HF arms do not depend on coarse grid and are reused as the same baseline, not duplicated independent trials.','reference_cost':'Common reference generation, imports, export and post-run test evaluation excluded and measured separately','selection':'Lowest validation relative L2 at60s; formal validation best; tests never select','statistics':'Five independent train pools plus optimization seeds, paired across arms. Shared validation and test set. Two budget checkpoints dependent.','random_execution_order_seed':64001,'hardware':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'threads':1,'sources':h,'scope':'Prospective short-construction-budget comparison; not 20minute convergence or crossPDE generalization'}
 if (ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==protocol
 else:write('protocol.json',protocol)
 wait_idle();ref=reference();validate(ref)
 jobs=[(a,lr) for a in ARMS for lr in protocol['learning_rates']];random.Random(64001).shuffle(jobs)
 for j,(a,lr) in enumerate(jobs):
  path=ROOT/'tuning'/f'{a}_lr{lr}'
  if (path/'result.json').exists():continue
  if path.exists():raise RuntimeError(f'Incomplete result retained; archive explicitly before retry: {path}')
  wait_idle();assert sources()==h;event('tuning',completed=j,total=len(jobs),arm=a,lr=lr)
  run(a,17,lr,60,path,ref,phase='tuning')
 chosen={}
 for a in ARMS:
  rs=[json.loads((ROOT/'tuning'/f'{a}_lr{lr}'/'result.json').read_text()) for lr in protocol['learning_rates']]
  r=min(rs,key=lambda x:(x['validation'],x['lr']));chosen[a]=r['lr']
 write('selected.json',chosen)
 jobs=[]
 # Randomized blocks preserve paired data pools while reducing execution-order confounding.
 rng=random.Random(64002)
 for seed in protocol['formal_seeds']:
  arms=ARMS.copy();rng.shuffle(arms);jobs.extend((a,seed) for a in arms)
 write('formal_manifest.json',jobs)
 for j,(a,seed) in enumerate(jobs):
  path=ROOT/'formal'/f'{a}_s{seed}'
  if (path/'evaluation.json').exists():continue
  wait_idle();assert sources()==h
  if not (path/'result.json').exists():
   if path.exists():raise RuntimeError(f'Incomplete result retained: {path}')
   event('formal',completed=j,total=len(jobs),arm=a,seed=seed,lr=chosen[a]);run(a,seed,chosen[a],180,path,ref)
  evaluate_run(path,ref)
 event('analysis');assert sources()==h
 import analyze;analyze.main()
 event('complete',tuning_runs=len(list((ROOT/'tuning').glob('*/result.json'))),formal_runs=len(list((ROOT/'formal').glob('*/result.json'))))
if __name__=='__main__':
 try:main()
 except Exception as e:
  event('failed',error=str(e),traceback=traceback.format_exc());raise
