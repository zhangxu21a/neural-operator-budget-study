import bootstrap
import scoring
from solver import parameters,solve
from model import SCALE
import numpy as np,pathlib,json,time,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
def prepare():
 p=ROOT/'fresh_reference.npz'
 if p.exists():return
 start=time.perf_counter();out={}
 for split,seed,count in [('test',39191,256),('ood',39291,128)]:
  params=parameters(seed,count)
  if split=='ood':params[:,:16]*=1.5
  out[split+'_p']=params;out[split+'_y']=np.stack([solve(a,128).ravel().astype('float32')/SCALE for a in params])
 np.savez_compressed(p,**out)
 (ROOT/'reference_manifest.json').write_text(json.dumps({'test_seed':39191,'test_count':256,'ood_seed':39291,'ood_count':128,'generation_seconds':time.perf_counter()-start,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'use':'evaluation only; never training or hyperparameter selection'},indent=2))
if __name__=='__main__':prepare()
