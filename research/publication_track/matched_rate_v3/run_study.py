import sys,pathlib,importlib.util,json,hashlib,time,os,random,subprocess,threading,fcntl,traceback
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;V2=R.parent/'architecture_horizon_v2'
spec=importlib.util.spec_from_file_location('_matched_rate_engine',V2/'study_engine.py');e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
a=e.load_file('_matched_rate_analysis',R/'analyze_study.py');assert pathlib.Path(a.__file__).resolve()==R/'analyze_study.py'
lock=(R/'.timing.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state={};stop=threading.Event()
def event(stage,**kw):
 global state
 state={'stage':stage,'time_unix':time.time(),'pid':os.getpid(),**kw};e.atomic_json(R/'status.json',state);print(json.dumps(state),flush=True)
def competing():
 found=[]
 for line in subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True).splitlines()[1:]:
  try:pid,cpu,comm=line.strip().split(None,2)
  except ValueError:continue
  if int(pid)!=os.getpid() and 'python' in comm.lower() and float(cpu)>30:
   cmd=subprocess.run(['ps','-p',pid,'-o','command='],capture_output=True,text=True).stdout.strip();found.append({'pid':int(pid),'cpu_percent':float(cpu),'command':cmd})
 return found
def idle():
 while True:
  other=competing()
  if not other:return
  event('waiting_for_other_compute',processes=other);time.sleep(15)
def monitor():
 with (R/'load_observations.jsonl').open('a',buffering=1) as f:
  while not stop.is_set():f.write(json.dumps({'time_unix':time.time(),'state':state.copy(),'competing':competing()})+'\n');stop.wait(5)
def hashes():
 old=json.loads((V2/'protocol.json').read_text())['sources'];files=[pathlib.Path(p) for p in old]+[R/'run_study.py',R/'analyze_study.py']
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def protocol():
 seeds=[2101,2202,2303,2404,2505];rng=random.Random(84002);jobs=[]
 for seed in seeds:
  block=[(seed,lr,arm) for lr in [.002,.008] for arm in ['coarse32','phys32']];rng.shuffle(block);jobs+=block
 return {'architecture':'fno','arms':['coarse32','phys32'],'rates':[.002,.008],'seeds':seeds,'parameter_seeds':[91000+s for s in seeds],'jobs':jobs,'budget':600,'checkpoints':[120,180,360,600],'pool':8192,'reference_seeds':[83001,83002,83003],'reference_counts':[32,128,64],'ood_multiplier':1.25,'gain':2,'nf':512,'nc':32,'input_output_grid':129,'new_tuning_trials':0,'common_initial_hf':32,'sources':hashes(),'primary_contrast':'corrected minus coarse within each fixed learning rate; both rates retained','secondary_contrast':'difference of correction contrasts across rates; within-trajectory horizon changes','retention':'All seeds, nonfinite loss flags and competing-load runs retained; no test-based rate selection or automatic replacements','timing':'Single thread, serial random blocks; 12-second reserve; real label costs; competing Python commands monitored every5s','scope':'Matching learning rate does not separate correction bias from correction cost and update count','engine_path':str(V2/'study_engine.py'),'analysis_path':str(R/'analyze_study.py')}
def references(p):
 f=R/'reference.npz'
 if f.exists():
  m=json.loads((R/'reference_manifest.json').read_text());assert hashlib.sha256(f.read_bytes()).hexdigest()==m['sha256'];return dict(e.np.load(f))
 start=time.perf_counter();out={};details={}
 for split,seed,n in zip(['validation','test','ood'],p['reference_seeds'],p['reference_counts']):
  ts=time.perf_counter();params=e.parameters(seed,n,'channel')
  if split=='ood':params[:,:16]*=p['ood_multiplier']
  oracle=e.Oracle(params);out[split+'_p']=params;out[split+'_x']=e.np.stack([e.make_input(q) for q in params])[:,None];out[split+'_y']=e.np.stack([oracle.get('hf',i) for i in range(n)])
  details[split]={'seconds':time.perf_counter()-ts,'max_hf_residual':max(oracle.residuals)};event('reference',split=split,count=n)
 e.np.savez_compressed(f,**out);e.atomic_json(R/'reference_manifest.json',{'seconds':time.perf_counter()-start,'splits':details,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'excluded_common_cost':True});return out
def execution_audit():
 observations=[json.loads(x) for x in (R/'load_observations.jsonl').read_text().splitlines()];rows=[]
 for f in sorted((R/'formal').glob('*/result.json')):
  r=json.loads(f.read_text());obs=[o for o in observations if r['started_at_unix']<=o['time_unix']<=r['started_at_unix']+r['wall']];flag=[o for o in obs if o['competing']]
  rows.append({'run':f.parent.name,'samples':len(obs),'flagged_samples':len(flag),'flagged_details':flag,'wall':r['wall'],'unattributed_wall':r['wall']-sum(r['timings'].values())})
 e.atomic_json(R/'execution_audit.json',{'runs':rows,'flagged_runs':sum(r['flagged_samples']>0 for r in rows),'scope':'Python observations cannot certify whole-host or thermal isolation; no flagged runs removed.'})
def main():
 p=protocol();f=R/'protocol.json'
 if f.exists():assert json.loads(f.read_text())==json.loads(json.dumps(p))
 else:e.atomic_json(f,p)
 event('starting',total=20);thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  idle();ref=references(p)
  for j,(seed,lr,arm) in enumerate(p['jobs']):
   d=R/'formal'/f'{arm}_lr{lr}_s{seed}'
   if (d/'evaluation.json').exists():continue
   idle();assert hashes()==p['sources']
   if not (d/'result.json').exists():
    if d.exists():raise RuntimeError(f'Incomplete output retained: {d}')
    event('formal',completed=j,total=20,arm=arm,lr=lr,seed=seed,budget=600)
    e.run('fno',arm,seed,lr,600,d,ref,phase='formal',pool=p['pool'],checkpoints=p['checkpoints'])
   event('evaluation',completed=j,total=20,arm=arm,lr=lr,seed=seed);e.evaluate_run(d,ref)
  event('analysis');assert hashes()==p['sources'];a.main(e);stop.set();thread.join();execution_audit()
  e.atomic_json(R/'completion.json',{'complete':True,'formal_runs':20,'budget_points':80,'completed_at_unix':time.time(),'pending_editorial':True});event('complete',formal_runs=20,budget_points=80)
 finally:stop.set();thread.join()
if __name__=='__main__':
 try:main()
 except Exception as exc:event('failed',error=str(exc),traceback=traceback.format_exc());raise
