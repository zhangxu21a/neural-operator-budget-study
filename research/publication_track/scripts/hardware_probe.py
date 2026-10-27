import os,sys,pathlib,time,json,statistics,platform
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R.parent/'mainline_revision'/'darcy_optimized'))
import scoring,torch,numpy as np
from fno_model import FNO,fine_inputs
from model import DeepONet,input_fields
from solver import parameters
OUT=R/'experiments'/'hardware_probe.json'
assert not OUT.exists()
torch.set_num_threads(1)
r={'torch':torch.__version__,'platform':platform.platform(),'cpu_threads':torch.get_num_threads(),'mps_available':torch.backends.mps.is_available(),'cuda_available':torch.cuda.is_available(),'purpose':'synchronized optimizer-step microbenchmark, NOT end-to-end training comparison','warmup_steps':3,'timed_steps':10,'cases':[]}
p=parameters(52001,8)
for arch in ['fno','deeponet']:
 for device in ['cpu','mps']:
  c={'arch':arch,'device':device}
  try:
   if device=='mps' and not r['mps_available']:raise RuntimeError('MPS unavailable')
   torch.manual_seed(123)
   net=(FNO() if arch=='fno' else DeepONet()).to(device)
   x=(fine_inputs(p) if arch=='fno' else input_fields(p)).to(device)
   y=torch.zeros((8,129**2),device=device);opt=torch.optim.AdamW(net.parameters(),lr=.008)
   times=[]
   for j in range(13):
    if device=='mps':torch.mps.synchronize()
    t=time.perf_counter();opt.zero_grad(set_to_none=True);loss=(net(x)-y).square().mean();loss.backward();opt.step()
    if device=='mps':torch.mps.synchronize()
    if j>=3:times.append(time.perf_counter()-t)
   c.update(status='passed',median_step_seconds=statistics.median(times),steps_seconds=times,loss=float(loss.detach().cpu()))
  except Exception as e:c.update(status='unsupported_or_failed',error=type(e).__name__+': '+str(e))
  r['cases'].append(c);print(json.dumps(c),flush=True)
OUT.write_text(json.dumps(r,indent=2))
