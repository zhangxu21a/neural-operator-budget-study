import runner
from runner import *
def evaluate_run(out):
 out=pathlib.Path(out)
 if (out/'evaluation.json').exists():return
 ck=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu');cfg=ck['config'];net=DeepONet() if cfg['arch']=='deeponet' else FNO();ref=dict(np.load(ROOT/'fresh_reference.npz'));old=np.load(BASE/'reference.npz');ref.update(validation_p=old['validation_p'],validation_y=old['validation_y']);result={}
 for budget,c in ck['checkpoints'].items():
  net.load_state_dict(c['model']);assert c['decision_ready']<=int(budget)
  metrics={}
  for split in ['validation','test','ood']:
   metric,_=evaluate(net,make_inputs(ref[split+'_p'],cfg['arch']),torch.tensor(ref[split+'_y']),batch=8);metrics[split]=metric
  assert abs(metrics['validation']['relative_l2']-c['validation'])<1e-7
  result[budget]=metrics|{k:v for k,v in c.items() if k not in ['model','validation']}
 (out/'evaluation.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({'evaluated':out.name,'test':{k:v['test']['relative_l2'] for k,v in result.items()}}),flush=True)
