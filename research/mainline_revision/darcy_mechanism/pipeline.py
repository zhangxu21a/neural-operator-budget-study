import pathlib,json,time,hashlib,random,traceback,subprocess,sys,os
ROOT=pathlib.Path(__file__).resolve().parent
ARMS=['data','gradient','uniform','field','pretrain','hf','phys_only','author_mlmc']
LRS=[.002,.004,.008,.016,.032,.064];SEEDS=[11,22,33]
MATCHED=['data','gradient','uniform','field','hf']
def save(name,data):
 p=ROOT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def choices(method):
 return [(lr,w) for lr in [.004,.008,.016] for w in [30,120]] if method=='pretrain' else [(lr,0) for lr in LRS]
def tuning_name(m,lr,w):return f'{m}_lr{lr}_warm{w}'
def formal_name(m,lr,seed,w):return f'{m}_lr{lr}_seed{seed}_warm{w}'
sources=['../darcy_optimized/vendor/mlmc_optim/mlmc_optim/batcher.py','../darcy_optimized/vendor/mlmc_optim/mlmc_optim/gradient.py','../darcy_optimized/vendor/mlmc_optim/examples/examples_src/models/fno.py','runner.py','policies.py','bootstrap.py','pipeline.py','evaluate.py','prepare_reference.py','validate.py','post_validation.py']+['../darcy_optimized/'+f for f in ['scoring.py','fno_model.py','fno_scoring.py','author.py']]+['../darcy/'+f for f in ['solver.py','teacher.py','model.py','runtime.py']]
if not (ROOT/'protocol.json').exists():
 save('protocol.json',{'arms':ARMS,'seeds':SEEDS,'tuning_configs':{m:choices(m) for m in ARMS},'tuning_seconds':180,'formal_seconds':1200,'report_seconds':[300,1200],'matched_lr':.008,'matched_methods':MATCHED,'fixed_acquisition_rate':.25,'training_pool':2048,'test_seed':39191,'ood_seed':39291,'hardware':'Apple M4 single CPU thread, serial timed runs','mechanism':'gradient/uniform/field share fixed mean acquisition probability .25, teacher and IP-corrected target. Native data retains adaptive V/B cost allocation. Different score computation costs remain charged. Purchase counts can differ through unique cache and updates.','warmup':'pretrain tunes3 learning rates x2 warmup times, same6 trial allocation as others. Fixed absolute warmup, no optimizer/schedule reset.','budget':'common15second validation reserve;5min/20min checkpoints from same horizon-independent lr schedule','source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}})
p=json.loads((ROOT/'protocol.json').read_text())
for f,h in p['source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
import runner
from runner import run
from evaluate import evaluate_run
from prepare_reference import prepare
status={'pid':os.getpid(),'stage':'preparing','started':time.time(),'completed':[]};save('status.json',status)
try:
 prepare()
 jobs=[(m,lr,w) for m in ARMS for lr,w in choices(m)];random.Random(722).shuffle(jobs)
 for m,lr,w in jobs:
  out=ROOT/'tuning'/tuning_name(m,lr,w);status.update(stage='tuning',current=out.name);save('status.json',status)
  if not (out/'result.json').exists():run(m,'fno',101,lr,180,out,'tuning',warmup_seconds=w)
  status['completed'].append(str(out));save('status.json',status)
 selected={}
 for m in ARMS:
  rr=[json.loads((ROOT/'tuning'/tuning_name(m,lr,w)/'result.json').read_text()) for lr,w in choices(m)]
  winner=min(rr,key=lambda r:r['validation']);selected[m]={'lr':winner['lr'],'warmup_seconds':winner['warmup_seconds'],'allocated_search_seconds':1080,'actual_search_seconds':sum(r['wall'] for r in rr)}
 save('selected.json',selected)
 jobs=[]
 for m in ARMS:
  for seed in SEEDS:jobs.append((m,selected[m]['lr'],seed,selected[m]['warmup_seconds']))
 for m in MATCHED:
  for seed in SEEDS:
   job=(m,.008,seed,0)
   if job not in jobs:jobs.append(job)
 random.Random(723).shuffle(jobs);save('formal_manifest.json',{'jobs':jobs,'count':len(jobs),'reuse':'Identical selected and matched configurations share one trajectory, not independent repetitions.'})
 for m,lr,seed,w in jobs:
  out=ROOT/'formal'/formal_name(m,lr,seed,w);status.update(stage='formal',current=out.name);save('status.json',status)
  if not (out/'result.json').exists():run(m,'fno',seed,lr,1200,out,'formal',warmup_seconds=w)
  evaluate_run(out);status['completed'].append(str(out));save('status.json',status)
 status.update(stage='analysis');save('status.json',status)
 subprocess.run([sys.executable,str(ROOT/'summarize.py')],check=True)
 subprocess.run([sys.executable,str(ROOT/'post_validation.py')],check=True)
 subprocess.run([sys.executable,str(ROOT/'plot_results.py')],check=True)
 status.update(stage='complete',finished=time.time());save('status.json',status)
except BaseException as exc:
 status.update(stage='failed',error=repr(exc),traceback=traceback.format_exc());save('status.json',status);raise
