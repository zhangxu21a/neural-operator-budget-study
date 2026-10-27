import scoring
import torch,numpy as np
from author import FNO2d
from model import input_fields
from solver import coefficient
def fine_inputs(params):return torch.tensor(np.stack([np.log(coefficient(p,128)) for p in params])[:,None],dtype=torch.float32)
class FNO(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=FNO2d(8,8,32)
  self.last_features=None
  self.net.fc2.register_forward_pre_hook(self.capture)
 def capture(self,module,args):self.last_features=args[0]
 def forward(self,x,details=False):
  z=x if x.shape[-1]==129 else torch.nn.functional.interpolate(x,size=(129,129),mode='bilinear',align_corners=True)
  return self.at_resolution(z,details)
 def at_resolution(self,z,details=False):
  n=z.shape[-1];t=torch.linspace(0,1,n);xx,yy=torch.meshgrid(t,t,indexing='ij')
  inp=torch.cat([z,xx[None,None].expand(len(z),1,n,n),yy[None,None].expand(len(z),1,n,n)],dim=1)
  out=self.net(inp).flatten(1)
  if details:return out,self.last_features.reshape(len(z),-1,128)
  return out
