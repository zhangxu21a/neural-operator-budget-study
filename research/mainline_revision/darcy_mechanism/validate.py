import runner
from runner import *
import itertools

torch.set_num_threads(1)
torch.manual_seed(71)
params=parameters(30101,40);oracle=Oracle(params);teacher=Teacher(params)
coarse=np.stack([oracle.get_coarse(i).ravel()/SCALE for i in range(40)])
fine=np.stack([oracle.get_fine(i).ravel()/SCALE for i in range(40)])
teacher.fit(list(range(24)),fine[:24]-coarse[:24],list(range(24,32)),fine[24:32]-coarse[24:32])
net=FNO();x=make_inputs(params,'fno');ids=np.array([32,33])
pred,H=net(x[ids],details=True);mean,dev=teacher.predict(ids)
fm,fq=field_score(teacher,ids)
assert np.allclose(fm,mean,rtol=1e-5,atol=1e-6)
qref=np.mean(dev**2,axis=(0,2))+float(np.mean(teacher.audit_residual**2))*1e-4
assert np.allclose(fq,qref,rtol=2e-5,atol=1e-10)
assert np.allclose(teacher_mean(teacher,ids),mean,rtol=1e-5,atol=1e-6)
def grad(target):
 net.zero_grad(set_to_none=True);loss=.5*(net(x[ids])-torch.tensor(target,dtype=torch.float32)).square().mean();loss.backward()
 return torch.cat([(torch.view_as_real(p.grad).flatten() if p.grad.is_complex() else p.grad.flatten()) for p in net.parameters()]).detach().clone()
truth=grad(fine[ids]);cases=[]
for method in ['gradient','uniform','field']:
 if method=='gradient':m,q=score_fields(teacher,ids,H,True)
 elif method=='field':m,q=field_score(teacher,ids)
 else:m,q=teacher_mean(teacher,ids),np.ones(2)
 p=probabilities(q,np.ones(2),.25,pmin=.001);assert abs(p.mean()-.25)<1e-6
 expected=torch.zeros_like(truth);base=coarse[ids]+m
 for mask in itertools.product([0,1],repeat=2):
  z=np.array(mask);weight=np.prod(np.where(z,p,1-p));target=base+z[:,None]*(fine[ids]-base)/p[:,None]
  expected+=float(weight)*grad(target)
 err=float(torch.norm(expected-truth)/torch.norm(truth));assert err<2e-5,err
 cases.append({'method':method,'probabilities':p.tolist(),'mean_probability':float(p.mean()),'exact_gate_enumeration_gradient_relative_error':err})
(ROOT/'validation.json').write_text(json.dumps({'passed':True,'fresh_seed':30101,'field_score_matches_expanded_reference':True,'cases':cases},indent=2))
for method in ['data','gradient','uniform','field','pretrain','hf','phys_only','author_mlmc']:
 run(method,'fno',71,.008,30,ROOT/'smoke'/method,'tuning',pool=64,warmup_seconds=4)
print('ALL VALIDATIONS AND REAL TRAINING SMOKES PASSED',flush=True)
