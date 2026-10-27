"""Two predeclared timing diagnostics; no test evaluation or reselection."""
import core
from core import *
import subprocess,threading
D=ROOT/'timing_sensitivity'
def main():
 assert json.loads((ROOT/'status.json').read_text())['stage']=='complete'
 p=json.loads((ROOT/'timing_sensitivity_protocol.json').read_text())
 original=json.loads((ROOT/'protocol.json').read_text())
 assert all(hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h for f,h in original['sources'].items())
 selected_bytes=(ROOT/'selected.json').read_bytes()
 D.mkdir(exist_ok=False)
 (D/'source_manifest.json').write_text(json.dumps({str(pathlib.Path(__file__).resolve()):hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'timing_sensitivity_protocol.json').read_bytes()).hexdigest()},indent=2))
 data=np.load(ROOT/'reference.npz');ref={k:data[k] for k in ['validation_x','validation_y']};data.close();results=[];stop=threading.Event()
 def competing():
  found=[]
  for line in subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True).splitlines()[1:]:
   try:pid,cpu,cmd=line.strip().split(None,2)
   except ValueError:continue
   if int(pid)!=os.getpid() and 'python' in cmd.lower() and float(cpu)>30:found.append({'pid':int(pid),'cpu_percent':float(cpu)})
  return found
 def monitor():
  with (D/'load.jsonl').open('w',buffering=1) as f:
   while not stop.is_set():f.write(json.dumps({'time':time.time(),'competing':competing()})+'\n');stop.wait(5)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  for arm,lr in p['trials']:
   while competing():time.sleep(10)
   print('diagnostic',arm,lr,flush=True)
   old=json.loads((ROOT/'tuning'/f'{arm}_lr{lr}'/'result.json').read_text())
   r=run(arm,p['seed'],lr,p['budget_per_trial_seconds'],D/f'{arm}_lr{lr}',ref,phase='tuning',pool=p['pool'])
   others=[json.loads((ROOT/'tuning'/f'{arm}_lr{x}'/'result.json').read_text())['validation'] for x in original['learning_rates'] if x!=lr]
   results.append({'arm':arm,'lr':lr,'original_validation':old['validation'],'repeat_validation':r['validation'],'original_steps':old['steps'],'repeat_steps':r['steps'],'original_wall':old['wall'],'repeat_wall':r['wall'],'original_other_candidate_best_validation':min(others),'repeat_remains_below_other_original_candidates':r['validation']<min(others),'nonfinite':r['nonfinite_loss']})
  assert (ROOT/'selected.json').read_bytes()==selected_bytes
  (D/'summary.json').write_text(json.dumps({'results':results,'selected_unchanged':True,'no_test_evaluation':True,'additional_allocated_seconds':120,'purpose':p['purpose'],'all_sources_unchanged':all(hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h for f,h in original['sources'].items())},indent=2))
 finally:stop.set();thread.join()
 (D/'complete.json').write_text(json.dumps({'complete':True,'time_unix':time.time()}));print('complete',flush=True)
if __name__=='__main__':main()
