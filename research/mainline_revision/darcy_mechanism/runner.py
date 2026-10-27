import bootstrap
import scoring
import pathlib,time,json,copy,random
import numpy as np,torch
from solver import Oracle,parameters
from model import DeepONet,input_fields,evaluate,SCALE
from teacher import Teacher,probabilities
from scoring import score,cached_audit
from fno_model import FNO,fine_inputs
def make_inputs(params,arch):return input_fields(params) if arch=="deeponet" else fine_inputs(params)
from fno_scoring import score_fields,audit_stats
from policies import field_score,teacher_mean
from author import mlmc_gradient_estimate,MLMCBatcher
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent/'darcy'
class ResolutionAdapter(torch.nn.Module):
 def __init__(self,net):super().__init__();self.net=net
 def forward(self,x):return self.net.at_resolution(x)

def run(method,arch,seed,lr,budget,out,phase='formal',pool=2048,warmup_seconds=60):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed+19000)
 ref=np.load(BASE/'reference.npz');vx=make_inputs(ref['validation_p'],arch);vy=torch.tensor(ref['validation_y'])
 t0=time.perf_counter();elapsed=lambda:time.perf_counter()-t0
 params=parameters(1001 if phase=='formal' else 2001,pool);x=make_inputs(params,arch)
 net=DeepONet() if arch=='deeponet' else FNO()
 opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
 oracle=Oracle(params);teacher=Teacher(params);cache={};rng=np.random.default_rng(seed+9000);gate=np.random.default_rng(seed+19000)
 learned=method in ['data','gradient','uniform','field'];physical=method=='phys_only'
 costs={k:0. for k in ['coarse','fine','physics','teacher','scoring','training','validation','checkpoint']};fine_times=[]
 steps=0;rows=[];acquisition=[];next_fit=0;last_val=0;best=None;bestval=float('inf');bestready=0;beststeps=0;milestones={}
 def get(kind,i):
  i=int(i);start=time.perf_counter();known=i in getattr(oracle,'phys' if kind=='physics' else kind)
  val=getattr(oracle,'get_'+kind)(i);spent=time.perf_counter()-start;costs[kind]+=spent
  if kind=='fine' and not known:fine_times.append(spent)
  return val.ravel()/SCALE
 def base(i):return get('coarse',i)+(get('physics',i) if physical else 0)
 def fit():
  start=time.perf_counter();nested=costs['coarse']+costs['physics'];audit_spent=0.
  fitids=[i for i in sorted(oracle.fine) if not 24<=i<32];audit=list(range(24,32))
  y=np.stack([oracle.fine[i].ravel()/SCALE-base(i) for i in fitids]);ab=np.stack([base(i) for i in audit]);ay=np.stack([oracle.fine[i].ravel()/SCALE for i in audit])
  teacher.fit(fitids,y,audit,ay-ab);cache.update(base=ab,y=ay,delta=cached_audit(teacher,ab,ay))
  if arch=='fno' and method=='data':
   ts=time.perf_counter();cache['VB']=audit_stats(net,x,ab,ay,teacher);audit_spent=time.perf_counter()-ts;costs['scoring']+=audit_spent
  costs['teacher']+=time.perf_counter()-start-(costs['coarse']+costs['physics']-nested)-audit_spent
 for i in range(32):get('coarse',i);get('fine',i)
 if learned:fit();next_fit=100
 last_state=copy.deepcopy(net.state_dict());lastready=elapsed();laststeps=0
 cutoff=budget-min(15.,budget/2)
 short_cutoff=285 if budget>=1200 else None
 short_done=False
 batches=iter([])
 if method=='author_mlmc':
  assert arch=='fno'
  batcher=MLMCBatcher({'mlmc_pairing':'hierarchy'},[33,129],[pool,pool//2],[8,4],pool,seed)
  adapter=ResolutionAdapter(net);criterion=lambda a,b:.5*(a-b).square().mean()
 def assess():
  nonlocal best,bestval,bestready,beststeps,last_val
  ts=time.perf_counter();metric,_=evaluate(net,vx,vy,batch=8)
  if metric['relative_l2']<bestval:
   bestval=metric['relative_l2'];best=copy.deepcopy(net.state_dict());bestready=lastready;beststeps=laststeps
  costs['validation']+=time.perf_counter()-ts;last_val=elapsed()
  rows.append({'ready':last_val,'state_ready':lastready,'steps':laststeps,'validation':metric['relative_l2'],'fine':oracle.count['fine']})
 def capture(seconds):
  assert best is not None
  assert elapsed()<=seconds,(seconds,elapsed())
  milestones[str(seconds)]={'model':copy.deepcopy(best),'validation':bestval,'selected_state_ready':bestready,'selected_steps':beststeps,'decision_ready':elapsed(),'counts':copy.deepcopy(oracle.count),'timings':copy.deepcopy(costs),'steps_completed':laststeps}
 assess()  # A valid initial fallback also records divergent optimizer trials honestly.
 nonfinite_loss=False
 while elapsed()<cutoff:
  if short_cutoff and not short_done and elapsed()>=short_cutoff:
   assess();capture(300);short_done=True
  fraction=min(1.,elapsed()/cutoff)
  for group in opt.param_groups:group['lr']=lr/(1+elapsed()/60.)
  if learned and steps>=next_fit and len(oracle.fine)<pool:fit();next_fit=steps+100
  if method=='author_mlmc':
   try:batch=next(batches)
   except StopIteration:batches=iter(batcher.get_epoch_indices()[1]);batch=next(batches)
   ci=np.asarray(batch[0][1]);fi=np.asarray(batch[1][1])
   yc=torch.tensor(np.stack([get('coarse',i).reshape(129,129)[::4,::4].ravel() for i in ci]))
   yf=torch.tensor(np.stack([get('fine',i) for i in fi]))
   ycp=torch.tensor(np.stack([get('coarse',i).reshape(129,129)[::4,::4].ravel() for i in fi]))
   ts=time.perf_counter();xf=x[fi];xc=x[ci,:,::4,::4];xcp=x[fi,:,::4,::4]
   loss=mlmc_gradient_estimate(adapter,criterion,xc,yc)+mlmc_gradient_estimate(adapter,criterion,xf,yf,xcp,ycp)
  else:
   ids=np.unique(rng.integers(0,pool,size=8));ts=time.perf_counter()
   if learned:
    if arch=='deeponet':pred,h,tr=net(x[ids],details=True)
    else:pred,h=net(x[ids],details=True)
   else:pred=net(x[ids])
   costs['training']+=time.perf_counter()-ts
   target=np.empty((len(ids),129**2),dtype='float32');missing=[]
   for j,i in enumerate(ids):
    if method=='coarse' or (method=='pretrain' and elapsed()<warmup_seconds):target[j]=get('coarse',i)
    elif i in oracle.fine:target[j]=oracle.fine[i].ravel()/SCALE
    elif method in ['hf','pretrain']:target[j]=get('fine',i)
    elif method=='phys_only':target[j]=base(i)
    else:missing.append(j)
   if missing:
    mids=ids[missing];bs=np.stack([base(i) for i in mids]);ts=time.perf_counter()
    c0=max(.0001,sum(costs[k] for k in ['training','coarse','physics','teacher','scoring'])/max(8*steps,32));cf=max(float(np.median(fine_times[-64:])),1e-6)
    if arch=='deeponet':mean,q,p,diag=score(net,x,teacher,mids,h[missing],tr,cache['base'],cache['y'],c0,cf,True,cache['delta'])
    else:
     if method=='uniform':mean=teacher_mean(teacher,mids);q=np.ones(len(mids))
     elif method=='field':mean,q=field_score(teacher,mids)
     else:mean,q=score_fields(teacher,mids,h[missing],True)
     if method=='data':
      V,B=cache['VB'];rate=float(np.clip(np.sqrt(B*c0/(max(V-B,1e-12)*cf)),.001,1.))
     else:rate=.25
     p=probabilities(q,np.full(len(mids),cf),rate,pmin=.001)
    bs+=mean;costs['scoring']+=time.perf_counter()-ts;gates=gate.random(len(p))<p
    for k,(i,j) in enumerate(zip(mids,missing)):
     target[j]=bs[k]+(get('fine',i)-bs[k])/p[k] if gates[k] else bs[k]
     acquisition.append((steps,int(i),float(p[k]),bool(gates[k]),elapsed(),float(q[k]),float(rate)))
   ts=time.perf_counter();loss=.5*(pred-torch.from_numpy(target)).square().mean()
  if not bool(torch.isfinite(loss)):
   nonfinite_loss=True;net.load_state_dict(last_state);break
  opt.zero_grad(set_to_none=True);loss.backward();opt.step();costs['training']+=time.perf_counter()-ts
  steps+=1
  # Retain last valid update for hard wall-clock cutoff; no evaluation of overshooting state.
  if elapsed()>cutoff:net.load_state_dict(last_state);break
  ts=time.perf_counter();last_state=copy.deepcopy(net.state_dict());lastready=elapsed();laststeps=steps;costs['checkpoint']+=time.perf_counter()-ts
  if elapsed()-last_val>=max(10.,min(60.,budget/5)):assess()
 assess();capture(int(budget));wall=elapsed();assert wall<=budget
 config={'nonfinite_loss':nonfinite_loss,'method':method,'warmup_seconds':warmup_seconds if method=='pretrain' else 0,'rate_policy':'adaptive' if method=='data' else ('fixed .25' if learned else 'none'),'arch':arch,'seed':seed,'lr':lr,'phase':phase,'pool':pool,'allocated_wall':budget,'wall':wall,'counts':oracle.count,'timings':costs,'steps':laststeps,'validation':bestval,'checkpoint_policy':'validation best only, charged inside budget','milestones':{k:{kk:vv for kk,vv in v.items() if kk!='model'} for k,v in milestones.items()}}
 torch.save({'checkpoints':milestones,'config':config},out/'checkpoint.pt');(out/'result.json').write_text(json.dumps(config,indent=2));(out/'learning_curve.json').write_text(json.dumps(rows,indent=2));np.save(out/'acquisitions.npy',np.asarray(acquisition,dtype=float))
 print(json.dumps({'method':method,'warmup_seconds':warmup_seconds if method=='pretrain' else 0,'rate_policy':'adaptive' if method=='data' else ('fixed .25' if learned else 'none'),'arch':arch,'seed':seed,'budget':budget,'wall':wall,'steps':laststeps,'fine':oracle.count['fine'],'val':bestval}),flush=True)
 return config
