import scoring
import numpy as np,torch
from teacher import probabilities
# Orthonormal basis of the three-member ensemble's mean-zero subspace.
Q=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0,-2/np.sqrt(6)]],dtype='float32')
def score_fields(teacher,ids,features,fast=True):
 F=features.detach().numpy();N=F.shape[1]
 if fast:
  coef=np.einsum('bd,edr->ebr',teacher.features[ids],teacher.weights);cm=coef.mean(0)
  mean=cm@teacher.basis.T
  dev=(np.einsum('ek,ebr->kbr',Q,coef-cm[None]))@teacher.basis.T
  proj=np.einsum('kbn,bnh->kbh',dev,F)/N
  bias=dev.mean(-1)
  q=(np.sum(proj**2,axis=(0,2))+np.sum(bias**2,axis=0))/3
 else:
  mean,dev=teacher.predict(ids);proj=np.einsum('ebn,bnh->ebh',dev,F)/N
  q=np.mean(np.sum(proj**2,axis=2)+dev.mean(-1)**2,axis=0)
 # Common nonnegative calibration floor; no high-fidelity labels of queried inputs used.
 q+=float(np.mean(teacher.audit_residual**2))*1e-4
 return mean,q

def audit_stats(net,x,base,y,teacher):
 with torch.no_grad():
  ap,H=net(x[24:32],details=True);N=H.shape[1];H=torch.cat([H,torch.ones(8,N,1)],dim=-1)
  gf=torch.einsum('bn,bnh->bh',ap-torch.tensor(y),H)/N
  delta=base+teacher.predict(list(range(24,32)))[0]-y
  gd=torch.einsum('bn,bnh->bh',torch.tensor(delta),H)/N
  return float(gf.var(0,unbiased=True).sum()),float(gd.square().sum(1).mean())
