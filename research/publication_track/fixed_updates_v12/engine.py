"""Prespecified fixed-update diagnostic. Never modifies original computation sources."""
import os,sys,pathlib,json,time,hashlib,copy,random,subprocess,threading,fcntl,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent;P=ROOT.parent
sys.path.insert(0,str(P/'formal_mg_v1'))
import core as c
np,torch=c.np,c.torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def state(stage,**kw):
 v={'stage':stage,'pid':os.getpid(),'time':time.time(),**kw};dump(ROOT/'status.json',v);print(json.dumps(v),flush=True)
def check(p):
 for f,h in p['sources'].items():assert sha(f)==h, f
 assert sha(ROOT/'protocol.json')==(ROOT/'protocol.sha256').read_text().strip()
def tensorhash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def competing():
 ans=[]
 for line in subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True).splitlines()[1:]:
  a=line.strip().split(None,2)
  if len(a)==3 and int(a[0])!=os.getpid() and 'python' in a[2].lower() and float(a[1])>30:ans.append({'pid':int(a[0]),'cpu':float(a[1]),'name':a[2]})
 return ans
stop=threading.Event()
def monitor():
 with (ROOT/'load_observations.jsonl').open('a',buffering=1) as f:
  while not stop.is_set():f.write(json.dumps({'time':time.time(),'competing_python':competing()})+'\n');stop.wait(15)
def references(p):
 out=ROOT/'reference.npz';meta=ROOT/'reference_manifest.json'
 if meta.exists():assert sha(out)==json.loads(meta.read_text())['sha256'];return dict(np.load(out))
 assert not out.exists();ref={};cost={};start=time.perf_counter()
 for split,seed,n in zip(['validation','test','ood'],p['reference_seeds'],p['reference_counts']):
  t=time.perf_counter();params=c.parameters(seed,n,'channel')
  if split=='ood':params[:,:16]*=1.25
  oracle=c.Oracle(params);ref[split+'_p']=params;ref[split+'_x']=np.stack([c.make_input(x) for x in params])[:,None];ref[split+'_y']=np.stack([oracle.get('hf',i) for i in range(n)])
  cost[split]={'seconds':time.perf_counter()-t,'max_residual':max(oracle.residuals)};state('references',split=split)
 np.savez_compressed(out,**ref);dump(meta,{'seconds':time.perf_counter()-start,'splits':cost,'sha256':sha(out),'outside_training':True});return ref

def prepare(p,seed):
 out=ROOT/'data'/str(seed);meta=out/'manifest.json'
 if meta.exists():
  m=json.loads(meta.read_text());assert all(sha(out/f)==h for f,h in m['files'].items());return out,m
 assert not out.exists(),f'Incomplete data retained: {out}'
 out.mkdir(parents=True);start=time.perf_counter();params=c.parameters(p['parameter_seed_offset']+seed,p['pool'],'channel')
 rng=np.random.default_rng(p['batch_seed_offset']+seed);batches=[np.unique(rng.integers(0,p['pool'],8)) for _ in range(p['max_updates'])]
 needed=np.unique(np.concatenate([np.arange(32)]+batches));lookup={int(v):i for i,v in enumerate(needed)}
 padded=np.full((p['max_updates'],8),-1,dtype=np.int32);lengths=np.array([len(x) for x in batches],dtype=np.int32)
 for j,x in enumerate(batches):padded[j,:len(x)]=[lookup[int(i)] for i in x]
 np.save(out/'parameters.npy',params);np.save(out/'needed_ids.npy',needed);np.save(out/'batch_rows.npy',padded);np.save(out/'batch_lengths.npy',lengths)
 shape=(len(needed),129*129);X=np.lib.format.open_memmap(out/'inputs.npy',mode='w+',dtype='float32',shape=(len(needed),1,129,129));Y={arm:np.lib.format.open_memmap(out/(arm+'.npy'),mode='w+',dtype='float32',shape=shape) for arm in p['arms']}
 oracle=c.Oracle(params);t=time.perf_counter()
 for i in range(32):oracle.get('hf',i)
 hf_seconds=time.perf_counter()-t;input_seconds=0.
 for row,i in enumerate(needed):
  i=int(i);t=time.perf_counter();X[row,0]=c.make_input(params[i]);input_seconds+=time.perf_counter()-t
  for arm in p['arms']:
   kind='hf' if i<32 else arm;Y[arm][row]=oracle.get(kind,i);assert np.isfinite(Y[arm][row]).all()
   # Avoid retaining duplicate label buffers; common initial HF cache remains.
   if kind!='hf':del oracle.cache[kind][i]
  if (row+1)%256==0:state('precompute_labels',seed=seed,done=row+1,total=len(needed))
 X.flush()
 for a in Y.values():a.flush()
 torch.manual_seed(seed);net=c.FNO();initial_hash=tensorhash(net);torch.save(net.state_dict(),out/'initial_state.pt')
 hf_max_residual=max(oracle.residuals);hf_max_iterations=max(oracle.iterations)
 label_seconds=dict(oracle.times)
 del X,Y,oracle,net
 manifest={'seed':seed,'parameter_seed':p['parameter_seed_offset']+seed,'pool':p['pool'],'needed':len(needed),'input_seconds':input_seconds,'label_seconds':label_seconds,'common_initial_hf_seconds':hf_seconds,'hf_max_residual':hf_max_residual,'hf_max_iterations':hf_max_iterations,'wall_seconds':time.perf_counter()-start,'initial_state_hash':initial_hash,'files':{f.name:sha(f) for f in out.iterdir() if f.is_file() and not f.name.startswith('._')},'scope':'Only scheduled input IDs plus initial32; complete sequence fixed before training; all label work outside fixed-update run'}
 # Cached targets are generated once per pool and shared across rates; report rather than duplicate this expense.
 dump(meta,manifest);return out,manifest

def train(p,seed,rate,arm,ref,data,manifest):
 out=ROOT/'formal'/f'{arm}_lr{rate}_s{seed}';result=out/'result.json'
 if result.exists():return out
 assert not out.exists(),f'Incomplete training directory retained: {out}'
 out.mkdir(parents=True);started=time.time();t0=time.perf_counter();torch.manual_seed(seed);net=c.FNO();net.load_state_dict(torch.load(data/'initial_state.pt',weights_only=True));assert tensorhash(net)==manifest['initial_state_hash']
 opt=torch.optim.AdamW(net.parameters(),lr=rate,weight_decay=1e-4);X=np.load(data/'inputs.npy',mmap_mode='r');Y=np.load(data/(arm+'.npy'),mmap_mode='r');batches=np.load(data/'batch_rows.npy');lengths=np.load(data/'batch_lengths.npy')
 vx=torch.tensor(ref['validation_x']);vy=torch.tensor(ref['validation_y']);best=float('inf');beststep=0;beststate=None;curve=[];milestones={};timings={'training':0.,'validation':0.,'checkpoint':0.};failure=None
 def assess(step):
  nonlocal best,beststep,beststate
  t=time.perf_counter();metric,_=c.evaluate(net,vx,vy,batch=8);timings['validation']+=time.perf_counter()-t
  val=metric['relative_l2']
  if np.isfinite(val) and val<best:
   best=val;beststep=step;t=time.perf_counter();beststate=copy.deepcopy(net.state_dict());timings['checkpoint']+=time.perf_counter()-t
  curve.append({'step':step,'validation':val,'best_step':beststep,'elapsed':time.perf_counter()-t0})
 assess(0)
 assert beststate is not None, "Initial validation nonfinite; preserve failure"
 with (out/'updates.jsonl').open('w',buffering=1) as f:
  for step in range(1,p['max_updates']+1):
   ids=batches[step-1,:lengths[step-1]];lr=rate/(1+(step-1)/p['lr_tau_updates']);t=time.perf_counter()
   for g in opt.param_groups:g['lr']=lr
   x=torch.from_numpy(np.array(X[ids],copy=True));y=torch.from_numpy(np.array(Y[ids],copy=True));opt.zero_grad(set_to_none=True);loss=.5*(net(x)-y).square().mean()
   if not bool(torch.isfinite(loss)):
    failure={'reason':'nonfinite_loss','step':step};break
   loss.backward()
   if not all(bool(torch.isfinite(q.grad).all()) for q in net.parameters() if q.grad is not None):failure={'reason':'nonfinite_gradient','step':step};break
   opt.step();timings['training']+=time.perf_counter()-t
   f.write(json.dumps({'step':step,'lr':lr,'loss':float(loss.detach()),'batch_rows':ids.tolist()})+'\n')
   if step%p['validation_every']==0:
    assess(step);state('training',seed=seed,rate=rate,arm=arm,step=step,total=p['max_updates'],elapsed=time.perf_counter()-t0)
   if step in p['endpoints']:
    t=time.perf_counter();milestones[str(step)]={'last':copy.deepcopy(net.state_dict()),'selected':copy.deepcopy(beststate),'selected_step':beststep,'validation':best};timings['checkpoint']+=time.perf_counter()-t
 r={'seed':seed,'rate':rate,'arm':arm,'started_at_unix':started,'steps_completed':step if failure is None else step-1,'failure':failure,'wall_seconds':time.perf_counter()-t0,'timings':timings,'initial_state_hash':manifest['initial_state_hash'],'data_manifest_sha256':sha(data/'manifest.json'),'endpoints':{k:{'selected_step':v['selected_step'],'validation':v['validation']} for k,v in milestones.items()}}
 t=time.perf_counter();torch.save(milestones,out/'checkpoints.pt');r['export_seconds']=time.perf_counter()-t;dump(out/'curve.json',curve);dump(result,r);return out

def evaluate(out,ref):
 if (out/'evaluation.json').exists():return
 t=time.perf_counter();checkpoints=torch.load(out/'checkpoints.pt',weights_only=False,map_location='cpu');net=c.FNO();answer={}
 for step,ck in checkpoints.items():
  answer[step]={}
  for which in ['last','selected']:
   net.load_state_dict(ck[which]);answer[step][which]={}
   for split in ['test','ood']:
    metric,_=c.evaluate(net,torch.tensor(ref[split+'_x']),torch.tensor(ref[split+'_y']),batch=8);answer[step][which][split]=metric
 answer['offline_seconds']=time.perf_counter()-t;dump(out/'evaluation.json',answer)

def main():
 lock=(ROOT/'.run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=json.loads((ROOT/'protocol.json').read_text());check(p);thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  state('starting',trajectories=20);ref=references(p)
  for seed in p['seeds']:
   check(p);data,m=prepare(p,seed)
   for job in [x for x in p['jobs'] if x[0]==seed]:
    seed,rate,arm=job;check(p);out=train(p,seed,rate,arm,ref,data,m);state('offline_evaluation',seed=seed,rate=rate,arm=arm);evaluate(out,ref)
  check(p);state('analysis');subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)
  dump(ROOT/'completion.json',{'complete':True,'trajectories':20,'dependent_endpoints':40,'finished_at':time.time(),'pending_editorial':True});state('complete',trajectories=20)
 finally:stop.set();thread.join()
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=str(e),traceback=traceback.format_exc());raise
