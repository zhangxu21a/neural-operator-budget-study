import sys,pathlib,importlib.util,json,hashlib
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_architecture_engine',R/'study_engine.py');e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
def main():
 data=e.np.load(R.parent/'formal_mg_v1/reference.npz');ref={k:data[k][:8] for k in ['validation_x','validation_y']};data.close();report={'architectures':{},'runs':[]}
 for arch in ['fno','deeponet']:
  e.torch.manual_seed(71);net=e.make_model(arch);x=e.torch.tensor(ref['validation_x'][:2]);pred=net(x)
  assert pred.shape==(2,129*129) and e.torch.isfinite(pred).all()
  loss=pred.square().mean();loss.backward();assert all(e.torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
  report['architectures'][arch]={'parameters':sum(p.numel() for p in net.parameters()),'input_shape':list(x.shape),'output_shape':list(pred.shape)}
 for arch,arm,budget,caps in [('fno','coarse32',25,[15,25]),('fno','hf_online',25,[25]),('deeponet','phys32',25,[15,25]),('deeponet','hf128',45,[30,45])]:
  r=e.run(arch,arm,71,.002,budget,R/'preflight'/f'{arch}_{arm}',ref,phase='smoke',pool=256,checkpoints=caps)
  assert r['initial_complete'] and r['steps']>0 and not r['nonfinite_loss'] and r['wall']<=budget
  assert all(m['decision_ready']<=int(cap) for cap,m in r['milestones'].items())
  report['runs'].append({k:r[k] for k in ['architecture','arm','wall','steps','initial_complete','max_hf_residual']})
 report['source_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [R/'study_engine.py',R/'study_analysis.py',R/'study_pipeline.py']}
 report['passed']=True;e.atomic_json(R/'preflight.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
