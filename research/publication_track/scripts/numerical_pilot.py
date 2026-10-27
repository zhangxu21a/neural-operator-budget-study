"""Prospective numerical screen; no learned-model claims. Original files read-only."""
import os,sys,pathlib,json,time,hashlib,platform
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'mainline_revision'/'darcy'
sys.path.insert(0,str(BASE))
import solver
import numpy as np
from scipy.sparse.linalg import spsolve
OUT=ROOT/'experiments'/'numerical_pilot'
OUT.mkdir(exist_ok=False)
protocol={'seed':51001,'samples':6,'family':'channel','log_coefficient_gains':[1,2], 'fine_grids':[128,256,512], 'coarse_grids':[8,32], 'reference_grid':1024,'smoothing_steps_each_side':3,'omega':.65,'target_grid':128,'purpose':'prospective cost, correction bias and mesh-convergence screen; no training claims','timing':'serial CPU, single BLAS thread; includes assembly; no artificial waits','reference_caveat':'1024 is a numerical reference, not exact continuum solution','source_hash':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'solver_hash':hashlib.sha256((BASE/'solver.py').read_bytes()).hexdigest(),'platform':platform.platform()}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
params=solver.parameters(51001,6,'channel');np.save(OUT/'parameters.npy',params)
rows=[]
def coeff(p,n,g):return solver.coefficient(p,n)**g
def direct(p,n,g):
 t=time.perf_counter(); A=solver.matrix(coeff(p,n,g)); u=solver.field(spsolve(A,np.ones(A.shape[0])),n)
 return u,time.perf_counter()-t,A
def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
for gain in [1,2]:
 for idx,p in enumerate(params):
  reference,rt,_=direct(p,1024,gain);reference=reference[::8,::8]
  for nf in [128,256,512]:
   fine,ft,A=direct(p,nf,gain);hf=fine[::nf//128,::nf//128];u=fine[1:-1,1:-1].ravel()
   residual=float(np.linalg.norm(A@u-1)/np.sqrt(A.shape[0]))
   for nc in [8,32]:
    ts=time.perf_counter();coarse=solver.prolong(solver.field(spsolve(solver.matrix(coeff(p,nc,gain)),np.ones((nc-1)**2)),nc),nf);ct=time.perf_counter()-ts
    ts=time.perf_counter(); Acopy=solver.matrix(coeff(p,nf,gain));P=solver.interpolation(nc,nf)
    v=coarse[1:-1,1:-1].ravel();r=np.ones_like(v)-Acopy@v;dinv=1/Acopy.diagonal();e=np.zeros_like(v)
    for _ in range(3):e+=.65*dinv*(r-Acopy@e)
    Galerkin=(P.T@Acopy@P).tocsc();e+=P@spsolve(Galerkin,P.T@(r-Acopy@e))
    for _ in range(3):e+=.65*dinv*(r-Acopy@e)
    corrected=solver.field(v+e,nf);pt=time.perf_counter()-ts
    ec=(v+e)-u
    row={'gain':gain,'sample':idx,'nf':nf,'nc':nc,'fine_seconds':ft,'coarse_seconds':ct,'correction_seconds':pt,'reference_seconds':rt,'hf_residual':residual,'coarse_rel_hf':rel(coarse,fine),'corrected_rel_hf':rel(corrected,fine),'corrected_energy_rel_hf':float(np.sqrt((ec@(A@ec))/(u@(A@u)))),'fine_rel_reference':rel(hf,reference),'corrected_rel_reference':rel(corrected[::nf//128,::nf//128],reference),'coarse_rel_reference':rel(coarse[::nf//128,::nf//128],reference),'k_ratio':float(coeff(p,nf,gain).max()/coeff(p,nf,gain).min())}
    rows.append(row)
    with (OUT/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   del A,fine
  print(json.dumps({'gain':gain,'sample':idx,'rows':len(rows)}),flush=True)
(OUT/'result.json').write_text(json.dumps({'complete':True,'rows':rows},indent=2))
