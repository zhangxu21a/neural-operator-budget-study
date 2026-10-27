"""Direct versus Galerkin geometric-multigrid-preconditioned CG, including setup."""
import os,sys,pathlib,time,json,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'mainline_revision'/'darcy'))
import solver,numpy as np
from scipy.sparse.linalg import spsolve,splu,cg,LinearOperator
OUT=R/'experiments'/'solver_benchmark';OUT.mkdir(exist_ok=False)
(OUT/'protocol.json').write_text(json.dumps({'seed':51001,'samples':6,'family':'channel','gain':[1,2],'fine_grids':[128,512],'coarsest_grid':8,'jacobi_sweeps':3,'omega':.65,'rtol':1e-8,'atol':0,'maxiter':200,'costs_include':'coefficient evaluation, matrix assembly, preconditioner setup, solve','direct_repeats':'fresh direct solve per case, not reused pilot timing','order':'direct and MG first alternated by sample','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2))
def build_mg(A,n):
 mats=[A];Ps=[];Ds=[1/A.diagonal()]
 while n>8:
  P=solver.interpolation(n//2,n);Ps.append(P);A=(P.T@A@P).tocsr();mats.append(A);Ds.append(1/A.diagonal());n//=2
 coarse=splu(mats[-1].tocsc())
 def cycle(l,b):
  if l==len(mats)-1:return coarse.solve(b)
  A=mats[l];di=Ds[l];x=np.zeros_like(b)
  for _ in range(3):x+=.65*di*(b-A@x)
  x+=Ps[l]@cycle(l+1,Ps[l].T@(b-A@x))
  for _ in range(3):x+=.65*di*(b-A@x)
  return x
 return LinearOperator(mats[0].shape,matvec=lambda b:cycle(0,b),dtype=np.float64),len(mats)
rows=[]
for gain in [1,2]:
 for i,p in enumerate(solver.parameters(51001,6,'channel')):
  for n in [128,512]:
   vals={};metrics={};order=['direct','mgcg'] if i%2==0 else ['mgcg','direct']
   for method in order:
    ts=time.perf_counter();A=solver.matrix(solver.coefficient(p,n)**gain);assembly=time.perf_counter()-ts;b=np.ones(A.shape[0]);setup=0;iters=[]
    if method=='direct':start=time.perf_counter();x=spsolve(A,b);solve=time.perf_counter()-start;info=0
    else:
     start=time.perf_counter();M,levels=build_mg(A,n);setup=time.perf_counter()-start
     start=time.perf_counter();x,info=cg(A,b,M=M,rtol=1e-8,atol=0,maxiter=200,callback=lambda x:iters.append(1));solve=time.perf_counter()-start
    vals[method]=x;metrics[method]={'assembly_seconds':assembly,'setup_seconds':setup,'solve_seconds':solve,'total_seconds':assembly+setup+solve,'iterations':len(iters),'info':int(info),'relative_residual':float(np.linalg.norm(A@x-b)/np.linalg.norm(b))}
   err=float(np.linalg.norm(vals['direct']-vals['mgcg'])/np.linalg.norm(vals['direct']))
   row={'gain':gain,'sample':i,'nf':n,'order':order,'relative_difference':err,'passed':metrics['mgcg']['info']==0 and err<1e-6,'methods':metrics}
   rows.append(row)
   with (OUT/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps({'gain':gain,'sample':i,'nf':n,'passed':row['passed'],'direct_s':metrics['direct']['total_seconds'],'mgcg_s':metrics['mgcg']['total_seconds']}),flush=True)
(OUT/'result.json').write_text(json.dumps({'complete':True,'rows':rows,'all_passed':all(r['passed'] for r in rows)},indent=2))
