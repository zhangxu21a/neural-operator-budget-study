"""Prospective architecture/horizon study. All legacy loads are explicit and read-only."""
import os,sys,pathlib,time,json,copy,hashlib,platform,importlib.util
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent
V1=ROOT.parent/'formal_mg_v1'
RESEARCH=ROOT.parents[1]
def load_file(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module
mg=load_file('_publication_mg_numerics',V1/'core.py')
legacy=load_file('_publication_legacy_deeponet',RESEARCH/'mainline_revision/darcy/model.py')
torch,np,parameters,Oracle,make_input,evaluate=mg.torch,mg.np,mg.parameters,mg.Oracle,mg.make_input,mg.evaluate
def make_model(architecture):
 if architecture=='fno':return mg.FNO()
 if architecture=='deeponet':return legacy.DeepONet(n=128,rank=32)
 raise ValueError(architecture)
def atomic_json(path,data):
 path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def run(architecture,arm,seed,lr,budget,out,ref,phase='formal',pool=8192,checkpoints=None):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=False)
 # Reference tensors are common offline assets. All model and training inputs below are charged.
 vx=torch.tensor(ref['validation_x']);vy=torch.tensor(ref['validation_y'])
 started_at_unix=time.time();t0=time.perf_counter();elapsed=lambda:time.perf_counter()-t0
 torch.manual_seed(seed);params=parameters((91000 if phase=='formal' else 92000)+seed,pool,'channel');oracle=Oracle(params);net=make_model(architecture);opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
 rng=np.random.default_rng(seed+90000);xcache={};steps=0;beststeps=0;bestval=float('inf');best=None;bestready=0.;curve=[];milestones={};timings={'input':0.,'training':0.,'validation':0.,'checkpoint':0.};nonfinite=False
 def inputs(ids):
  ts=time.perf_counter()
  for i in ids:
   if int(i) not in xcache:xcache[int(i)]=make_input(params[i])
  x=torch.tensor(np.stack([xcache[int(i)] for i in ids])[:,None]);timings['input']+=time.perf_counter()-ts;return x
 def assess():
  nonlocal best,bestval,beststeps,bestready
  ts=time.perf_counter();old_ckpt=timings['checkpoint'];metric,_=evaluate(net,vx,vy,batch=8)
  if np.isfinite(metric['relative_l2']) and metric['relative_l2']<bestval:
   bestval=metric['relative_l2'];beststeps=steps;bestready=elapsed();tc=time.perf_counter();best=copy.deepcopy(net.state_dict());timings['checkpoint']+=time.perf_counter()-tc
  timings['validation']+=time.perf_counter()-ts-(timings['checkpoint']-old_ckpt)
  curve.append({'decision_ready':elapsed(),'steps':steps,'validation':metric['relative_l2'],'hf':len(oracle.cache['hf'])})
  atomic_json(out/'progress.json',{'architecture':architecture,'arm':arm,'seed':seed,'budget':budget,**curve[-1]})
 def capture(cap):
  tc=time.perf_counter();model=copy.deepcopy(best);timings['checkpoint']+=time.perf_counter()-tc
  if elapsed()>cap:raise RuntimeError(f'Budget exceeded: {elapsed()} > {cap}')
  milestones[str(cap)]={'model':model,'validation':bestval,'selected_steps':beststeps,'decision_ready':elapsed(),'counts':{k:len(v) for k,v in oracle.cache.items()},'timings':oracle.times|timings,'steps_completed':steps}
 # Initial assessed model provides a fallback even if a configuration exhausts time buying labels.
 assess();desired=128 if arm=='hf128' else 32
 for i in range(desired):
  if elapsed()>=budget-12:break
  oracle.get('hf',i)
 initial_complete=len(oracle.cache['hf'])==desired
 checkpoints=list(checkpoints or [budget]);assert checkpoints==sorted(set(checkpoints)) and checkpoints[-1]==budget;nextidx=0;lastval=elapsed();acquisition_log=[]
 while elapsed()<budget-12 and initial_complete:
  cap=checkpoints[nextidx]
  if elapsed()>=cap-12:
   assess();capture(cap);nextidx+=1
   if nextidx>=len(checkpoints):break
  size=desired if arm in ['hf32','hf128'] else pool
  ids=np.unique(rng.integers(0,size,8));x=inputs(ids);targets=[]
  for i in ids:
   kind='hf' if arm.startswith('hf') or int(i) in oracle.cache['hf'] else arm
   targets.append(oracle.get(kind,i))
  if elapsed()>=min(budget,checkpoints[nextidx])-6:
   # A completed acquisition may be charged without an optimizer update at a boundary.
   if checkpoints[nextidx]<budget:assess();capture(checkpoints[nextidx]);nextidx+=1;continue
   break
  ts=time.perf_counter()
  for group in opt.param_groups:group['lr']=lr/(1+elapsed()/60)
  opt.zero_grad(set_to_none=True);loss=.5*(net(x)-torch.tensor(np.stack(targets))).square().mean()
  if not bool(torch.isfinite(loss)):nonfinite=True;break
  loss.backward();opt.step();timings['training']+=time.perf_counter()-ts;steps+=1
  if elapsed()-lastval>=30:assess();lastval=elapsed()
 if not nonfinite:assess()
 # The last valid best remains valid if an attempted update diverged.
 for cap in checkpoints:
  if str(cap) not in milestones:capture(cap)
 wall=elapsed();result={'started_at_unix':started_at_unix,'architecture':architecture,'arm':arm,'seed':seed,'train_parameter_seed':(91000 if phase=='formal' else 92000)+seed,'lr':lr,'phase':phase,'pool':pool,'budget':budget,'wall':wall,'nonfinite_loss':nonfinite,'initial_complete':initial_complete,'steps':steps,'validation':bestval,'counts':{k:len(v) for k,v in oracle.cache.items()},'timings':oracle.times|timings,'max_hf_residual':max(oracle.residuals,default=0),'max_hf_iterations':max(oracle.iterations,default=0),'milestones':{k:{a:b for a,b in v.items() if a!='model'} for k,v in milestones.items()}}
 # Offline export and held-out evaluation are separately measured, never used in configuration choice.
 ts=time.perf_counter();torch.save({'config':result,'milestones':milestones},out/'checkpoint.pt');result['export_seconds']=time.perf_counter()-ts
 (out/'curve.json').write_text(json.dumps(curve,indent=2))
 (out/'result.json').write_text(json.dumps(result,indent=2))
 return result

def evaluate_run(out,ref):
 out=pathlib.Path(out);ts=time.perf_counter();ck=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu');net=make_model(ck['config']['architecture']);result={}
 for cap,c in ck['milestones'].items():
  net.load_state_dict(c['model']);result[cap]={}
  for split in ['test','ood']:
   metric,_=evaluate(net,torch.tensor(ref[split+'_x']),torch.tensor(ref[split+'_y']),batch=8)
   assert np.isfinite(metric['individual']).all(),(out,cap,split)
   result[cap][split]=metric
 result['offline_seconds']=time.perf_counter()-ts;atomic_json(out/'evaluation.json',result)
 return result
