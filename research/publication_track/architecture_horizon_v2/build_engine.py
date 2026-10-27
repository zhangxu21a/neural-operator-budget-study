from pathlib import Path
R=Path(__file__).resolve().parent
old=(R.parent/'formal_mg_v1/core.py').read_text()
header='''"""Prospective architecture/horizon study. All legacy loads are explicit and read-only."""
import os,sys,pathlib,time,json,copy,hashlib,platform,importlib.util
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent
V1=ROOT.parent/'formal_mg_v1'
RESEARCH=ROOT.parents[1]
def load_file(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module
mg=load_file('_publication_mg_numerics',V1/'core.py')
legacy=load_file('_publication_legacy_deeponet',RESEARCH/'mainline_revision/darcy/model.py')
torch,np,parameters,Oracle,make_input,evaluate=mg.torch,mg.np,mg.parameters,mg.Oracle,mg.make_input,mg.evaluate
def make_model(architecture):
 if architecture=='fno':return mg.FNO()
 if architecture=='deeponet':return legacy.DeepONet(n=128,rank=32)
 raise ValueError(architecture)
def atomic_json(path,data):
 path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
'''
s=old[old.index('def run('):old.index('\ndef evaluate_run')]
s=s.replace("def run(arm,seed,lr,budget,out,ref,phase='formal',pool=8192):", "def run(architecture,arm,seed,lr,budget,out,ref,phase='formal',pool=8192,checkpoints=None):")
s=s.replace('(71000 if phase==\'formal\' else 81000)', '(91000 if phase==\'formal\' else 92000)')
s=s.replace('net=FNO()', 'net=make_model(architecture)')
s=s.replace("checkpoints=([120] if budget>120 else [])+[budget]", "checkpoints=list(checkpoints or [budget]);assert checkpoints==sorted(set(checkpoints)) and checkpoints[-1]==budget")
s=s.replace("'arm':arm,'seed':seed", "'architecture':architecture,'arm':arm,'seed':seed")
s=s.replace("  curve.append({'decision_ready':elapsed(),'steps':steps,'validation':metric['relative_l2'],'hf':len(oracle.cache['hf'])})", "  curve.append({'decision_ready':elapsed(),'steps':steps,'validation':metric['relative_l2'],'hf':len(oracle.cache['hf'])})\n  atomic_json(out/'progress.json',{'architecture':architecture,'arm':arm,'seed':seed,'budget':budget,**curve[-1]})")
a=s.index(' def capture(cap):');b=s.index(' # Initial assessed',a)
s=s[:a]+''' def capture(cap):
  tc=time.perf_counter();model=copy.deepcopy(best);timings['checkpoint']+=time.perf_counter()-tc
  if elapsed()>cap:raise RuntimeError(f'Budget exceeded: {elapsed()} > {cap}')
  milestones[str(cap)]={'model':model,'validation':bestval,'selected_steps':beststeps,'decision_ready':elapsed(),'counts':{k:len(v) for k,v in oracle.cache.items()},'timings':oracle.times|timings,'steps_completed':steps}
'''+s[b:]
tail='''
def evaluate_run(out,ref):
 out=pathlib.Path(out);ts=time.perf_counter();ck=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu');net=make_model(ck['config']['architecture']);result={}
 for cap,c in ck['milestones'].items():
  net.load_state_dict(c['model']);result[cap]={}
  for split in ['test','ood']:
   metric,_=evaluate(net,torch.tensor(ref[split+'_x']),torch.tensor(ref[split+'_y']),batch=8)
   assert np.isfinite(metric['individual']).all(),(out,cap,split)
   result[cap][split]=metric
 result['offline_seconds']=time.perf_counter()-ts;atomic_json(out/'evaluation.json',result)
 return result
'''
(R/'study_engine.py').write_text(header+s+tail)
print(R/'study_engine.py')
