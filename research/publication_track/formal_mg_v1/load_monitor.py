"""Observational timing audit; never changes frozen experiment or controls other jobs."""
import os,json,time,subprocess,pathlib
root=pathlib.Path(__file__).resolve().parent
with (root/'load_observations.jsonl').open('a',buffering=1) as f:
 while True:
  s=json.loads((root/'status.json').read_text());others=[]
  for line in subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True).splitlines()[1:]:
   try:p,c,cmd=line.strip().split(None,2)
   except ValueError:continue
   if int(p) not in [os.getpid(),s['pid']] and 'python' in cmd.lower() and float(c)>30:others.append({'pid':int(p),'cpu_percent':float(c)})
  f.write(json.dumps({'observed_at':time.time(),'status':s,'other_python_load':others})+'\n')
  if s['stage'] in ['failed','complete']:break
  time.sleep(5)
