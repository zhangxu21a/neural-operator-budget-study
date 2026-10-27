import runtime
import torch
from torch import nn
import numpy as np
from solver import coefficient

SCALE=.05

def input_fields(params):
    return torch.tensor(np.stack([np.log(coefficient(p,32)) for p in params])[:,None],dtype=torch.float32)

class DeepONet(nn.Module):
    def __init__(self,n=128,rank=32):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,8,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(8,16,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(256,64),nn.GELU())
        self.branch=nn.Linear(64,rank)
        self.trunk=nn.Sequential(nn.Linear(10,64),nn.Tanh(),nn.Linear(64,rank))
        t=torch.linspace(0,1,n+1);x,y=torch.meshgrid(t,t,indexing='ij')
        coords=torch.stack([x,y,torch.sin(torch.pi*x),torch.sin(torch.pi*y),torch.cos(torch.pi*x),torch.cos(torch.pi*y),torch.sin(2*torch.pi*x),torch.sin(2*torch.pi*y),torch.cos(2*torch.pi*x),torch.cos(2*torch.pi*y)],dim=-1).reshape(-1,10)
        self.register_buffer('coords',coords)
        self.register_buffer('boundary',(16*x*(1-x)*y*(1-y)).reshape(-1,1))
    def forward(self,x,details=False):
        h=self.encoder(x);tr=self.trunk(self.coords)*self.boundary
        pred=self.branch(h)@tr.T/np.sqrt(tr.shape[1])
        return (pred,h,tr/np.sqrt(tr.shape[1])) if details else pred

@torch.no_grad()
def evaluate(net,x,y,batch=16):
    vals=[];preds=[]
    for i in range(0,len(x),batch):
        p=net(x[i:i+batch]);yy=y[i:i+batch]
        vals.extend((torch.linalg.vector_norm(p-yy,dim=1)/torch.linalg.vector_norm(yy,dim=1).clamp_min(1e-8)).tolist());preds.append(p.numpy())
    return {'relative_l2':float(np.mean(vals)),'p95':float(np.quantile(vals,.95)),'individual':vals},np.concatenate(preds)
