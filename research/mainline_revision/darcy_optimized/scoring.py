"""Exact algebraic factorization; no score-frequency or estimator change."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent/'darcy'))
import runtime
import numpy as np
import torch
from teacher import probabilities

def cached_audit(teacher,base,y):
    # Identical addition/subtraction ordering to the old per-step evaluation.
    return np.asarray(base+teacher.predict(list(range(24,32)))[0]-y,dtype='float32')

def score(net,x,teacher,ids,h,tr,audit_base,audit_y,c0,cf,fast=False,audit_delta=None):
    T=tr.detach().numpy();hh=h.detach().numpy();N=T.shape[0]
    if fast:
        coef=np.einsum('bd,edr->ebr',teacher.features[ids],teacher.weights)
        cm=coef.mean(axis=0)
        mean=cm@teacher.basis.T
        projections=(coef-cm[None])@(teacher.basis.T@T)/N
    else:
        mean,dev=teacher.predict(ids);projections=dev@T/N
    cal=teacher.audit_residual@T/N
    q=(np.mean(np.sum(projections**2,axis=2),axis=0)+np.mean(np.sum(cal**2,axis=1)))*(np.sum(hh**2,axis=1)+1)
    with torch.no_grad():
        if fast:
            ah=net.encoder(x[24:32]);at=tr.detach();ap=net.branch(ah)@at.T
            delta=audit_delta
        else:
            ap,ah,at=net(x[24:32],details=True)
            delta=audit_base+teacher.predict(list(range(24,32)))[0]-audit_y
        H=torch.cat([ah,torch.ones(8,1)],dim=1)
        G=((ap-torch.tensor(audit_y))@at/N)[:,:,None]*H[:,None,:]
        D=(torch.tensor(delta)@at/N)[:,:,None]*H[:,None,:]
        V=float(G.flatten(1).var(dim=0,unbiased=True).sum())
        B=float(D.square().flatten(1).sum(dim=1).mean())
        rate=float(np.clip(np.sqrt(B*max(c0,.0001)/(max(V-B,1e-12)*max(cf,1e-6))),.001,1.))
    p=probabilities(q,np.full(len(ids),cf),rate,pmin=.001)
    return mean,q,p,{'V':V,'B':B,'rate':rate}
