import os,sys,json,pathlib
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
MANIFEST=pathlib.Path('/Volumes/MacShare/Something/research_q1_20261024/new_directions/rl_experiments/runtime_manifest.json')
sys.path[:0]=list(json.loads(MANIFEST.read_text())['archive_paths'].values())
