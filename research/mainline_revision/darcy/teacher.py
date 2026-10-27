import runtime
import numpy as np

class Teacher:
    """Matched residual ridge ensemble. Physics changes the offset, not capacity.
    Eight common, purchased audit labels are excluded from ridge fitting.
    Features and output basis are fixed without unseen HF labels.
    """
    def __init__(self,params,n=128,ridge=1.):
        self.features=np.concatenate([np.ones((len(params),1)),params,params**2],axis=1).astype('float32')
        t=np.linspace(0,1,n+1);x,y=np.meshgrid(t,t,indexing='ij')
        raw=np.stack([(np.sin(np.pi*i*x)*np.sin(np.pi*j*y)).ravel() for i in range(1,5) for j in range(1,5)],axis=1)
        self.basis=np.linalg.qr(raw)[0].astype('float32')
        self.ridge=ridge;self.weights=np.zeros((3,41,16),dtype='float32')
        self.audit_residual=np.zeros((8,(n+1)**2),dtype='float32')
        self.rng=np.random.default_rng(101)
    def fit(self,ids,targets,audit_ids,audit_targets):
        X=self.features[ids];Y=targets@self.basis
        for j in range(3):
            w=self.rng.exponential(size=len(ids)).astype('float32')
            self.weights[j]=np.linalg.solve(X.T@(w[:,None]*X)+self.ridge*np.eye(X.shape[1]),X.T@(w[:,None]*Y))
        means=self.predict(audit_ids)[0]
        self.audit_residual=audit_targets-means
    def predict(self,ids):
        coef=np.einsum('bd,edr->ebr',self.features[ids],self.weights)
        fields=coef@self.basis.T
        return fields.mean(axis=0),fields-fields.mean(axis=0)[None]

def probabilities(q,cost,rate,pmin=.05):
    q=np.maximum(q,1e-12);cost=np.maximum(cost,1e-8)
    if rate>=1:return np.ones_like(q)
    if rate<=pmin+1e-12:return np.full_like(q,pmin)
    target=max(rate,pmin)*cost.sum()
    lo=0.;hi=1.
    calc=lambda v:np.clip(np.sqrt(q/(v*cost)),pmin,1.)
    while np.dot(calc(hi),cost)>target:hi*=4
    for _ in range(35):
        m=(lo+hi)/2
        if np.dot(calc(m),cost)>target:lo=m
        else:hi=m
    return calc(hi)
