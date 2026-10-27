"""Frozen MG budget study: read-only legacy solver and FNO imports."""
import os,sys,pathlib,time,json,copy,hashlib,platform
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent
RESEARCH=ROOT.parents[1]
sys.path.insert(0,str(RESEARCH/'mainline_revision/darcy_optimized'))
import scoring,torch,numpy as np
from solver import parameters,coefficient,matrix,field,prolong,interpolation
from scipy.sparse.linalg import spsolve,splu,cg,LinearOperator
from fno_model import FNO
from model import SCALE,evaluate
torch.set_num_threads(1)
ARMS=['coarse8','coarse32','phys8','phys32','hf_online','hf32','hf128']
def make_input(p):return np.log(coefficient(p,128)).astype('float32')*2
class Oracle:
 def __init__(self,params):
  self.params=params;self.cache={k:{} for k in ['hf','coarse8','coarse32','phys8','phys32']};self.times={k:0. for k in self.cache};self.iterations=[];self.residuals=[]
  self.Ps={n:interpolation(n//2,n) for n in [512,256,128,64,32,16]};self.Pc={n:interpolation(n,512) for n in [8,32]}
 def mg(self,A):
  mats=[A];Ps=[];Ds=[1/A.diagonal()];n=512
  while n>8:
   P=self.Ps[n];Ps.append(P);A=(P.T@A@P).tocsr();mats.append(A);Ds.append(1/A.diagonal());n//=2
  cf=splu(mats[-1].tocsc())
  def cycle(l,b):
   if l==len(mats)-1:return cf.solve(b)
   A=mats[l];x=np.zeros_like(b)
   for _ in range(3):x+=.65*Ds[l]*(b-A@x)
   x+=Ps[l]@cycle(l+1,Ps[l].T@(b-A@x))
   for _ in range(3):x+=.65*Ds[l]*(b-A@x)
   return x
  return LinearOperator(mats[0].shape,matvec=lambda b:cycle(0,b),dtype=np.float64)
 def get(self,kind,i):
  i=int(i)
  if i in self.cache[kind]:return self.cache[kind][i]
  t=time.perf_counter();p=self.params[i]
  if kind=='hf':
   A=matrix(coefficient(p,512)**2);b=np.ones(A.shape[0]);M=self.mg(A);iters=[]
   u,info=cg(A,b,M=M,rtol=1e-8,atol=0,maxiter=200,callback=lambda u:iters.append(1));res=float(np.linalg.norm(A@u-b)/np.linalg.norm(b))
   self.iterations.append(len(iters));self.residuals.append(res)
   if info or res>1.1e-8:raise RuntimeError(f'HF solver failure: info={info}, residual={res}')
   result=field(u,512)[::4,::4]
  else:
   nc=int(kind[4:] if kind.startswith('phys') else kind[6:]);Ac=matrix(coefficient(p,nc)**2);coarse=prolong(field(spsolve(Ac,np.ones(Ac.shape[0])),nc),512)
   if kind.startswith('phys'):
    A=matrix(coefficient(p,512)**2);v=coarse[1:-1,1:-1].ravel();r=np.ones_like(v)-A@v;e=np.zeros_like(v);D=1/A.diagonal();P=self.Pc[nc]
    for _ in range(3):e+=.65*D*(r-A@e)
    e+=P@spsolve((P.T@A@P).tocsc(),P.T@(r-A@e))
    for _ in range(3):e+=.65*D*(r-A@e)
    result=field(v+e,512)[::4,::4]
   else:result=coarse[::4,::4]
  result=result.ravel().astype('float32')/SCALE;self.cache[kind][i]=result;self.times[kind]+=time.perf_counter()-t
  return result

def run(arm,seed,lr,budget,out,ref,phase='formal',pool=8192):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=False)
 # Reference tensors are common offline assets. All model and training inputs below are charged.
 vx=torch.tensor(ref['validation_x']);vy=torch.tensor(ref['validation_y'])
 t0=time.perf_counter();elapsed=lambda:time.perf_counter()-t0
 torch.manual_seed(seed);params=parameters((71000 if phase=='formal' else 81000)+seed,pool,'channel');oracle=Oracle(params);net=FNO();opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
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
 def capture(cap):
  if elapsed()>cap:raise RuntimeError(f'Budget exceeded: {elapsed()} > {cap}')
  milestones[str(cap)]={'model':copy.deepcopy(best),'validation':bestval,'selected_steps':beststeps,'decision_ready':elapsed(),'counts':{k:len(v) for k,v in oracle.cache.items()},'timings':oracle.times|timings,'steps_completed':steps}
 # Initial assessed model provides a fallback even if a configuration exhausts time buying labels.
 assess();desired=128 if arm=='hf128' else 32
 for i in range(desired):
  if elapsed()>=budget-12:break
  oracle.get('hf',i)
 initial_complete=len(oracle.cache['hf'])==desired
 checkpoints=([120] if budget>120 else [])+[budget];nextidx=0;lastval=elapsed();acquisition_log=[]
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
 wall=elapsed();result={'arm':arm,'seed':seed,'train_parameter_seed':(71000 if phase=='formal' else 81000)+seed,'lr':lr,'phase':phase,'pool':pool,'budget':budget,'wall':wall,'nonfinite_loss':nonfinite,'initial_complete':initial_complete,'steps':steps,'validation':bestval,'counts':{k:len(v) for k,v in oracle.cache.items()},'timings':oracle.times|timings,'max_hf_residual':max(oracle.residuals,default=0),'max_hf_iterations':max(oracle.iterations,default=0),'milestones':{k:{a:b for a,b in v.items() if a!='model'} for k,v in milestones.items()}}
 # Offline export and held-out evaluation are separately measured, never used in configuration choice.
 ts=time.perf_counter();torch.save({'config':result,'milestones':milestones},out/'checkpoint.pt');result['export_seconds']=time.perf_counter()-ts
 (out/'curve.json').write_text(json.dumps(curve,indent=2))
 (out/'result.json').write_text(json.dumps(result,indent=2))
 return result

def evaluate_run(out,ref):
 out=pathlib.Path(out);ts=time.perf_counter();ck=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu');net=FNO();result={}
 for cap,c in ck['milestones'].items():
  net.load_state_dict(c['model']);result[cap]={}
  for split in ['test','ood']:
   metric,_=evaluate(net,torch.tensor(ref[split+'_x']),torch.tensor(ref[split+'_y']),batch=8);result[cap][split]=metric
 result['offline_seconds']=time.perf_counter()-ts;(out/'evaluation.json').write_text(json.dumps(result,indent=2))
 return result
