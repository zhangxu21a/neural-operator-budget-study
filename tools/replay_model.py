"""Optional fresh-run launcher; NOT executed during release preparation."""
import argparse,sys,types,os,importlib.util,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--study',choices=['formal_mg_v1','architecture_horizon_v2','matched_rate_v3'],required=True);ap.add_argument('--run',required=True);ap.add_argument('--output',type=Path);ap.add_argument('--train',action='store_true');a=ap.parse_args();root=a.root.resolve();p=root/'research/publication_track';f=p/a.study/'formal'/a.run/'result.json';r=json.loads(f.read_text())
if not a.train:print(json.dumps(r,indent=2));raise SystemExit(0)
if a.output is None or a.output.exists():raise SystemExit('Training requires --output pointing to a NEW directory.')
# Portability adapter bypasses the archived machine-specific runtime.py only.
# All numerical/model sources remain byte-identical; new timings are machine dependent.
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True;sys.modules['runtime']=types.ModuleType('runtime')
engine=p/('formal_mg_v1/core.py' if a.study=='formal_mg_v1' else 'architecture_horizon_v2/study_engine.py')
spec=importlib.util.spec_from_file_location('_release_engine',engine);e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
ref=dict(e.np.load(p/a.study/'reference.npz'));args=[r['arm'],r['seed'],r['lr'],r['budget'],a.output,ref]
if a.study!='formal_mg_v1':args.insert(0,r['architecture'])
kw={'phase':'formal','pool':r['pool']}
if a.study!='formal_mg_v1':kw['checkpoints']=sorted(map(int,r['milestones']))
e.run(*args,**kw)
print('Fresh training finished. Test evaluation is intentionally separate; archived results untouched.')
