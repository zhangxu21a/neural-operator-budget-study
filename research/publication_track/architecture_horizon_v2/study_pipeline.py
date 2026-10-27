"""Frozen serial study. Explicit module loading; unique local output only."""
import sys,pathlib,importlib.util,json,hashlib,time,os,random,subprocess,threading,fcntl,traceback,shutil
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_architecture_engine',ROOT/'study_engine.py');e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
analysis=e.load_file('_architecture_analysis',ROOT/'study_analysis.py')
assert pathlib.Path(analysis.__file__).resolve()==ROOT/'study_analysis.py'
lock=(ROOT/'.timing.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={};stop=threading.Event()
def event(stage,**kw):
 global state
 state={'stage':stage,'time_unix':time.time(),'pid':os.getpid(),**kw};e.atomic_json(ROOT/'status.json',state);print(json.dumps(state),flush=True)
def other_compute():
 rows=[]
 for line in subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True).splitlines()[1:]:
  try:pid,cpu,comm=line.strip().split(None,2)
  except ValueError:continue
  if int(pid)!=os.getpid() and 'python' in comm.lower() and float(cpu)>30:
   command=subprocess.run(['ps','-p',pid,'-o','command='],capture_output=True,text=True).stdout.strip()
   rows.append({'pid':int(pid),'cpu_percent':float(cpu),'command':command})
 return rows
def wait_idle():
 while True:
  other=other_compute()
  if not other:return
  event('waiting_for_other_compute',processes=other);time.sleep(15)
def monitor():
 with (ROOT/'load_observations.jsonl').open('a',buffering=1) as f:
  while not stop.is_set():
   f.write(json.dumps({'time_unix':time.time(),'stage':state.copy(),'competing':other_compute()})+'\n');stop.wait(5)
def source_hashes():
 base=json.loads((ROOT.parent/'formal_mg_v1/protocol.json').read_text())['sources']
 files=[pathlib.Path(p) for p in base]+[ROOT/x for x in ['study_engine.py','study_analysis.py','study_pipeline.py','preflight.py']]
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def protocol():
 arms={'deeponet':['coarse32','phys32','hf_online','hf32','hf128'],'fno':['coarse32','phys32','hf_online','hf128']};seeds=[1101,1202,1303,1404,1505];rates={'deeponet':[.0005,.002,.008],'fno':[.002,.008,.032]}
 tune=[(a,m,lr) for a,ms in arms.items() for m in ms for lr in rates[a]];random.Random(74001).shuffle(tune)
 jobs=[];rng=random.Random(74002)
 for seed in seeds:
  block=[(a,m,seed) for a,ms in arms.items() for m in ms];rng.shuffle(block);jobs+=block
 return {'arms':arms,'seeds':seeds,'formal_parameter_seeds':[91000+s for s in seeds],'tuning_seed':1017,'tuning_parameter_seed':93017,'rates':rates,'tuning_seconds':60,'tuning_budget_per_arm':180,'formal_seconds':{'deeponet':180,'fno':600},'checkpoints':{'deeponet':[120,180],'fno':[120,180,360,600]},'pool':8192,'gain':2,'fine_grid':512,'coarse_grid':32,'observation_grid':129,'hf_common_initial':32,'hf_fixed_subsets':[32,128],'reference_seeds':[73001,73002,73003],'reference_counts':[32,128,64],'ood_gaussian_multiplier':1.25,'tuning_jobs':tune,'formal_jobs':jobs,'sources':source_hashes(),'architecture_parameters':json.loads((ROOT/'preflight.json').read_text())['architectures'],'selection':'Validation only; same frozen lr schedule eta0/(1+t/60); no long-horizon reselection','scope':'New independent pools and evaluation sets; fixed-policy architecture/horizon replication, not capacity-matched ranking or convergence proof','budget_exclusions':'Imports/startup, shared reference generation, serialization and post-run tests excluded and measured separately','timing_policy':'Single CPU thread, serial trials, wait for other high-load Python before runs; log full commands every5s during runs. Retain and flag any competing-load trial, no automatic scientific reruns.','retention':'No seed replacement, no deletion of nonfinite or adverse outcomes; incomplete directories cause failure and remain in place','analysis_path':str(ROOT/'study_analysis.py'),'threads':1,'hardware':e.platform.platform(),'torch':e.torch.__version__}
def references(p):
 f=ROOT/'reference.npz'
 if f.exists():
  m=json.loads((ROOT/'reference_manifest.json').read_text());assert hashlib.sha256(f.read_bytes()).hexdigest()==m['sha256'];return dict(e.np.load(f))
 ts=time.perf_counter();out={};costs={}
 for split,seed,n in zip(['validation','test','ood'],p['reference_seeds'],p['reference_counts']):
  start=time.perf_counter();params=e.parameters(seed,n,'channel')
  if split=='ood':params[:,:16]*=p['ood_gaussian_multiplier']
  oracle=e.Oracle(params);out[split+'_p']=params;out[split+'_x']=e.np.stack([e.make_input(q) for q in params])[:,None];out[split+'_y']=e.np.stack([oracle.get('hf',i) for i in range(n)])
  costs[split]={'seconds':time.perf_counter()-start,'max_residual':max(oracle.residuals),'max_iterations':max(oracle.iterations)};event('reference',split=split,count=n)
 e.np.savez_compressed(f,**out);e.atomic_json(ROOT/'reference_manifest.json',{'seconds':time.perf_counter()-ts,'splits':costs,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'budget_exclusion':True});return out
def execution_audit():
 observations=[json.loads(x) for x in (ROOT/'load_observations.jsonl').read_text().splitlines()];rows=[]
 for phase in ['tuning','formal']:
  for f in sorted((ROOT/phase).glob('*/result.json')):
   r=json.loads(f.read_text());start=r['started_at_unix'];end=start+r['wall'];obs=[o for o in observations if start<=o['time_unix']<=end];flag=[o for o in obs if o['competing']]
   rows.append({'phase':phase,'run':f.parent.name,'observations':len(obs),'flagged_observations':len(flag),'competing_details':[o for o in flag],'wall':r['wall'],'unattributed_wall':r['wall']-sum(r['timings'].values())})
 e.atomic_json(ROOT/'execution_audit.json',{'runs':rows,'flagged_runs':sum(r['flagged_observations']>0 for r in rows),'caveat':'Five-second Python process observations do not certify whole-host or thermal isolation. No flagged run removed.'})
def main():
 pre=json.loads((ROOT/'preflight.json').read_text());assert pre['passed']
 for f,h in pre['source_hashes'].items():assert hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h
 p=protocol();path=ROOT/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(p))
 else:e.atomic_json(path,p)
 baseline=ROOT/'baseline_manuscript'
 if not baseline.exists():shutil.copytree(ROOT.parent/'manuscript/paper',baseline)
 h=p['sources'];event('starting',tuning_total=27,formal_total=45);thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  wait_idle();ref=references(p)
  for j,(arch,arm,lr) in enumerate(p['tuning_jobs']):
   out=ROOT/'tuning'/f'{arch}_{arm}_lr{lr}'
   if (out/'result.json').exists():continue
   if out.exists():raise RuntimeError(f'Incomplete output retained: {out}')
   wait_idle();assert source_hashes()==h;event('tuning',completed=j,total=27,architecture=arch,arm=arm,lr=lr)
   e.run(arch,arm,p['tuning_seed'],lr,60,out,ref,phase='tuning',pool=p['pool'])
  selected={}
  for arch,arms in p['arms'].items():
   selected[arch]={}
   for arm in arms:
    rr=[json.loads((ROOT/'tuning'/f'{arch}_{arm}_lr{lr}'/'result.json').read_text()) for lr in p['rates'][arch]]
    selected[arch][arm]=min(rr,key=lambda r:(r['validation'],r['lr']))['lr']
  e.atomic_json(ROOT/'selected.json',selected)
  for j,(arch,arm,seed) in enumerate(p['formal_jobs']):
   out=ROOT/'formal'/f'{arch}_{arm}_s{seed}'
   if (out/'evaluation.json').exists():continue
   wait_idle();assert source_hashes()==h
   if not (out/'result.json').exists():
    if out.exists():raise RuntimeError(f'Incomplete output retained: {out}')
    event('formal',completed=j,total=45,architecture=arch,arm=arm,seed=seed,budget=p['formal_seconds'][arch])
    e.run(arch,arm,seed,selected[arch][arm],p['formal_seconds'][arch],out,ref,phase='formal',pool=p['pool'],checkpoints=p['checkpoints'][arch])
   event('evaluation',completed=j,total=45,architecture=arch,arm=arm,seed=seed);e.evaluate_run(out,ref)
  event('analysis');assert source_hashes()==h;analysis.main(e)
  stop.set();thread.join();execution_audit()
  e.atomic_json(ROOT/'completion.json',{'complete':True,'tuning_runs':27,'formal_runs':45,'budget_points':130,'completed_at_unix':time.time(),'pending_editorial':'Interpretation review and final manuscript compilation'});event('complete',tuning_runs=27,formal_runs=45)
 finally:stop.set();thread.join()
if __name__=='__main__':
 try:main()
 except Exception as exc:event('failed',error=str(exc),traceback=traceback.format_exc());raise
