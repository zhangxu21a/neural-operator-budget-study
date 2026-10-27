"""Prospective feasibility experiment. One seed, fixed LR, no superiority inference."""
import os,sys,pathlib,time,json,hashlib,copy
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'mainline_revision'/'darcy_optimized'))
import scoring,torch,numpy as np
from solver import parameters,coefficient,matrix,field,prolong,interpolation,Oracle
from scipy.sparse.linalg import spsolve
from fno_model import FNO
from model import SCALE,evaluate
OUT=R/'experiments'/'training_pilot';OUT.mkdir(exist_ok=False)
torch.set_num_threads(1)
class Labels:
 def __init__(self,p,g,nc,nf=512):
  self.p=p;self.g=g;self.nc=nc;self.nf=nf;self.cache={k:{} for k in ['coarse','fine','physics']};self.cost={k:0. for k in self.cache};self.P=interpolation(nc,nf)
 def get(self,kind,i):
  if i in self.cache[kind]:return self.cache[kind][i]
  t=time.perf_counter();p=self.p[i];nf=self.nf
  if kind=='coarse':
   A=matrix(coefficient(p,self.nc)**self.g);u=prolong(field(spsolve(A,np.ones(A.shape[0])),self.nc),nf)
  elif kind=='fine':
   A=matrix(coefficient(p,nf)**self.g);u=field(spsolve(A,np.ones(A.shape[0])),nf)
  else:
   # Coarse lookup already charged separately, including when requested lazily.
   before=self.cost['coarse'];coarse=self.get('coarse',i);nested=self.cost['coarse']-before
   A=matrix(coefficient(p,nf)**self.g);v=coarse[1:-1,1:-1].ravel().astype('float64');r=np.ones_like(v)-A@v;e=np.zeros_like(v);di=1/A.diagonal()
   for _ in range(3):e+=.65*di*(r-A@e)
   Ac=(self.P.T@A@self.P).tocsc();e+=self.P@spsolve(Ac,self.P.T@(r-A@e))
   for _ in range(3):e+=.65*di*(r-A@e)
   u=field(v+e,nf)
  self.cache[kind][i]=u.astype('float32');self.cost[kind]+=time.perf_counter()-t-(nested if kind=='physics' else 0)
  return self.cache[kind][i]
 def target(self,kind,i):return self.get(kind,int(i))[::self.nf//128,::self.nf//128].ravel()/SCALE
# Independent adaptation consistency check on original mesh and coefficient.
p=parameters(53000,1);a=Labels(p,1,32,128);b=Oracle(p)
errs={k:float(np.max(np.abs(a.get(k,0)-(b.get_coarse(0)+b.get_physics(0) if k=='physics' else getattr(b,'get_'+k)(0))))) for k in ['coarse','fine','physics']}
assert max(errs.values())<1e-6
protocol={'status':'prospective feasibility pilot','seeds':[11],'lr':.008,'budget_seconds':90,'pool':256,'validation_size':16,'test_size':32,'initial_hf':32,'arms':['coarse','hf','phys_only'],'scenarios':[{'name':'resolved','gain':1,'nc':32,'nf':512},{'name':'stress','gain':2,'nc':8,'nf':512}],'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'adaptation_max_absolute_errors':errs,'training_seed':53001,'validation_seed':53002,'test_seed':53003,'scope':'Extremes chosen after numerical screen; no isolated-factor or method superiority claim. Shared reference generation and post-run test evaluation excluded from per-run online budget and logged separately. No hyperparameter selection. Randomized serial arm order. Labels solved on 512 grid and restricted to 129 output; does not establish full-resolution operator accuracy.'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
def inputs(p,g):return torch.tensor(np.stack([np.log(coefficient(q,128))*g for q in p])[:,None],dtype=torch.float32)
for scenario,g,nc in [('resolved',1,32),('stress',2,8)]:
 tp=parameters(53001,256,'channel');vp=parameters(53002,16,'channel');ep=parameters(53003,32,'channel')
 start=time.perf_counter();reference=Labels(np.concatenate([vp,ep]),g,nc);yy=np.stack([reference.target('fine',i) for i in range(48)]);preparation=time.perf_counter()-start
 vx=inputs(vp,g);ex=inputs(ep,g);vy=torch.tensor(yy[:16]);ey=torch.tensor(yy[16:])
 np.savez_compressed(OUT/(scenario+'_reference.npz'),validation_p=vp,test_p=ep,validation_y=yy[:16],test_y=yy[16:])
 (OUT/(scenario+'_reference_cost.json')).write_text(json.dumps({'wall_seconds':preparation,'fine_seconds':reference.cost['fine']},indent=2));del reference
 order=list(np.random.default_rng(53004+g).permutation(['coarse','hf','phys_only']))
 for method in order:
  path=OUT/(scenario+'_'+method);path.mkdir();torch.manual_seed(11);rng=np.random.default_rng(90011)
  t0=time.perf_counter();elapsed=lambda:time.perf_counter()-t0
  x=inputs(tp,g);net=FNO();opt=torch.optim.AdamW(net.parameters(),lr=.008,weight_decay=1e-4);oracle=Labels(tp,g,nc)
  for i in range(32):oracle.get('coarse',i);oracle.get('fine',i)
  best=copy.deepcopy(net.state_dict());bestval=float('inf');steps=0;traincost=0.;valcost=0.;lastval=-100.;curve=[];beststeps=0
  def assess():
   global best,bestval,beststeps,valcost,lastval
   t=time.perf_counter();metric,_=evaluate(net,vx,vy,batch=8)
   if metric['relative_l2']<bestval:bestval=metric['relative_l2'];best=copy.deepcopy(net.state_dict());beststeps=steps
   valcost+=time.perf_counter()-t;lastval=elapsed();curve.append({'wall':lastval,'steps':steps,'validation':metric['relative_l2']})
  assess()
  while elapsed()<75:
   ids=np.unique(rng.integers(0,256,8));target=[]
   for i in ids:
    kind='coarse' if method=='coarse' else ('fine' if i in oracle.cache['fine'] or method=='hf' else 'physics')
    target.append(oracle.target(kind,i))
   if elapsed()>=80:break
   for group in opt.param_groups:group['lr']=.008/(1+elapsed()/60)
   ts=time.perf_counter();opt.zero_grad(set_to_none=True);loss=.5*(net(x[ids])-torch.tensor(np.stack(target))).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss retained as failed pilot')
   loss.backward();opt.step();traincost+=time.perf_counter()-ts;steps+=1
   if elapsed()-lastval>=15:assess()
  assess();wall=elapsed();assert wall<90
  torch.save(best,path/'checkpoint.pt');net.load_state_dict(best)
  test,_=evaluate(net,ex,ey,batch=8)
  result={'scenario':scenario,'method':method,'seed':11,'budget':90,'wall':wall,'steps':steps,'selected_steps':beststeps,'validation':bestval,'test':test,'timings':oracle.cost|{'training':traincost,'validation':valcost},'counts':{k:len(v) for k,v in oracle.cache.items()},'order':order}
  (path/'result.json').write_text(json.dumps(result,indent=2));(path/'curve.json').write_text(json.dumps(curve,indent=2));print(json.dumps({k:result[k] for k in ['scenario','method','wall','steps','validation','counts']}),flush=True)
(OUT/'complete.json').write_text(json.dumps({'complete':True,'runs':6}))
