"""Local-only device plumbing fixes; preserves original archived sources."""
import os,sys,pathlib,time,json,statistics,copy
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'mainline_revision'/'darcy_optimized'))
import scoring,torch,numpy as np
from fno_model import FNO,fine_inputs
from model import DeepONet,input_fields
from solver import parameters
class DeviceFNO(FNO):
 def at_resolution(self,z,details=False):
  n=z.shape[-1];t=torch.linspace(0,1,n,device=z.device);xx,yy=torch.meshgrid(t,t,indexing='ij');inp=torch.cat([z,xx[None,None].expand(len(z),1,n,n),yy[None,None].expand(len(z),1,n,n)],dim=1);out=self.net(inp).flatten(1)
  return (out,self.last_features.reshape(len(z),-1,128)) if details else out
class ExactPool(torch.nn.Module):
 def forward(self,x):
  h,w=x.shape[-2:];rows=[]
  for i in range(4):
   cells=[]
   for j in range(4):cells.append(x[...,i*h//4:((i+1)*h+3)//4,j*w//4:((j+1)*w+3)//4].mean((-2,-1),keepdim=True))
   rows.append(torch.cat(cells,-1))
  return torch.cat(rows,-2)
torch.set_num_threads(1);p=parameters(52001,8);result={'purpose':'device-plumbing feasibility and synchronized microbenchmark; synthetic zero targets; NOT full training or accuracy comparison','cases':[]}
for arch in ['fno','deeponet']:
 torch.manual_seed(123);original=FNO() if arch=='fno' else DeepONet();fixed=DeviceFNO() if arch=='fno' else DeepONet();fixed.load_state_dict(original.state_dict())
 if arch=='deeponet':fixed.encoder[4]=ExactPool()
 x=fine_inputs(p) if arch=='fno' else input_fields(p)
 with torch.no_grad():cpuerr=float((original(x)-fixed(x)).abs().max())
 assert cpuerr<1e-6
 for device in ['cpu','mps']:
  c={'arch':arch,'device':device,'cpu_forward_max_abs_difference':cpuerr}
  try:
   net=copy.deepcopy(fixed).to(device);xx=x.to(device);y=torch.zeros((8,129**2),device=device);opt=torch.optim.AdamW(net.parameters(),lr=.008);times=[]
   for j in range(13):
    if device=='mps':torch.mps.synchronize()
    t=time.perf_counter();opt.zero_grad(set_to_none=True);loss=(net(xx)-y).square().mean();loss.backward();opt.step()
    if device=='mps':torch.mps.synchronize()
    if j>=3:times.append(time.perf_counter()-t)
   c.update(status='passed',median_step_seconds=statistics.median(times),steps_seconds=times,final_loss=float(loss.detach().cpu()))
  except Exception as e:c.update(status='unsupported_or_failed',error=type(e).__name__+': '+str(e))
  result['cases'].append(c);print(json.dumps(c),flush=True)
out=R/'experiments'/'hardware_portability_probe.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
