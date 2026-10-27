"""-div(k grad u)=1 on unit square, homogeneous Dirichlet boundary.
Conservative nodal flux stencil, harmonic face coefficient, independent grids.
"""
import runtime
import numpy as np
from scipy.sparse import diags,kron,csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

def parameters(seed, count, family='mixed'):
    rng=np.random.default_rng(seed)
    p=rng.normal(size=(count,20))
    p[:,16:]=rng.uniform(size=(count,4))
    if family=='smooth':p[:,19]=0.
    if family=='channel':p[:,19]=1.
    return p

def coefficient(p,n):
    t=np.linspace(0,1,n+1);x,y=np.meshgrid(t,t,indexing='ij')
    z=np.zeros_like(x)
    for j in range(16):
        a=j//4+1;b=j%4+1
        z+=p[j]*np.cos(np.pi*a*x)*np.cos(np.pi*b*y)/(a*a+b*b)**.75
    z=1.25*z
    if p[19]>.5:
        center=.25+.5*p[16]+.12*np.sin(2*np.pi*x+6*p[17])
        # Smooth high-contrast channel; avoids alignment artifacts from discontinuous masks.
        width=.025+.035*p[18]
        z+=2.5*np.exp(-((y-center)/width)**2)
    return np.exp(np.clip(z,-4,4))

def matrix(k):
    n=k.shape[0]-1;m=n-1;c=k[1:-1,1:-1]
    harm=lambda a,b:2*a*b/(a+b)
    xm=harm(c,k[:-2,1:-1])*n*n;xp=harm(c,k[2:,1:-1])*n*n
    ym=harm(c,k[1:-1,:-2])*n*n;yp=harm(c,k[1:-1,2:])*n*n
    diag=(xm+xp+ym+yp).ravel()
    up=-yp.ravel()[:-1].copy();up[np.arange(len(up))%m==m-1]=0
    dn=-ym.ravel()[1:].copy();dn[np.arange(len(dn))%m==m-1]=0
    return diags([diag,up,dn,-xp.ravel()[:-m],-xm.ravel()[m:]], [0,1,-1,m,-m],format='csr')

def field(vec,n):
    u=np.zeros((n+1,n+1));u[1:-1,1:-1]=vec.reshape(n-1,n-1);return u

def solve(p,n=128):
    A=matrix(coefficient(p,n));return field(spsolve(A,np.ones(A.shape[0])),n)

def prolong(u,n):
    nc=len(u)-1;t=np.linspace(0,1,n+1);x,y=np.meshgrid(t,t,indexing='ij')
    return RegularGridInterpolator((np.linspace(0,1,nc+1),)*2,u)((x,y))

def interpolation(nc,nf):
    x=np.arange(1,nf)/nf*nc;lo=np.floor(x).astype(int);w=x-lo
    rows=[];cols=[];vals=[]
    for i,(j,v) in enumerate(zip(lo,w)):
        if 0<j<nc:rows.append(i);cols.append(j-1);vals.append(1-v)
        if 0<j+1<nc:rows.append(i);cols.append(j);vals.append(v)
    P=csr_matrix((vals,(rows,cols)),shape=(nf-1,nc-1))
    return kron(P,P,format='csr')

class Oracle:
    def __init__(self,params,nc=32,nf=128):
        self.params=params;self.nc=nc;self.nf=nf;self.P=interpolation(nc,nf)
        self.coarse={};self.fine={};self.phys={};self.A={};self.count={'coarse':0,'fine':0,'physics':0}
    def get_coarse(self,i):
        if i not in self.coarse:
            u=solve(self.params[i],self.nc)
            self.coarse[i]=prolong(u,self.nf).astype('float32')
            self.count['coarse']+=1
        return self.coarse[i]
    def get_fine(self,i):
        if i not in self.fine:
            A=self.A.get(i)
            if A is None:A=matrix(coefficient(self.params[i],self.nf))
            self.fine[i]=field(spsolve(A,np.ones(A.shape[0])),self.nf).astype('float32')
            self.count['fine']+=1
        return self.fine[i]
    def get_physics(self,i):
        if i not in self.phys:
            v=self.get_coarse(i)[1:-1,1:-1].ravel().astype('float64')
            A=matrix(coefficient(self.params[i],self.nf));self.A[i]=A
            r=np.ones(A.shape[0])-A@v;dinv=1/A.diagonal();e=np.zeros_like(v)
            for _ in range(3):e+=.65*dinv*(r-A@e)
            # Galerkin two-grid defect correction: same operator as HF solve.
            Ac=(self.P.T@A@self.P).tocsc()
            e+=self.P@spsolve(Ac,self.P.T@(r-A@e))
            for _ in range(3):e+=.65*dinv*(r-A@e)
            self.phys[i]=field(e,self.nf).astype('float32');self.count['physics']+=1
        return self.phys[i]

def validate():
    p=np.zeros(20);p[19]=0
    n=32;A=matrix(coefficient(p,n));t=np.linspace(0,1,n+1)
    x,y=np.meshgrid(t,t,indexing='ij');exact=np.sin(np.pi*x)*np.sin(np.pi*y)
    b=2*np.pi*np.pi*exact[1:-1,1:-1].ravel();u=field(spsolve(A,b),n)
    err=np.linalg.norm(u-exact)/np.linalg.norm(exact)
    assert err<.001
    assert np.max(np.abs((A-A.T).data),initial=0)<1e-12
    return {'manufactured_relative_error_n32':float(err),'symmetry_max_error':float(np.max(np.abs((A-A.T).data),initial=0))}
