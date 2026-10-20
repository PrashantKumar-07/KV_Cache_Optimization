"""Durable serial continuation of reviewed experiments; no favorable retries."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((BASE/'queue_plan.json').read_text())
assert not (BASE/'queue_status.json').exists(), 'Inspect prior queue; do not duplicate'
status={'state':'waiting_for_quiet_timing','plan_sha256':sha(BASE/'queue_plan.json'),'completed_stages':[],'pid':os.getpid()}
def save():
 status['updated_utc']=datetime.now(timezone.utc).isoformat()
 p=BASE/'queue_status.tmp';p.write_text(json.dumps(status,indent=2)+'\n');os.replace(p,BASE/'queue_status.json')
try:
 while True:
  p=ROOT/'results/audit_20261016_timing/timing.json'
  if p.exists():
   r=json.loads(p.read_text())
   if r.get('execution_complete'):break
   if r.get('error'):raise RuntimeError('Quiet timing failed; inspect preserved evidence before continuing')
  try:
   st=Path(f"/proc/{plan['timing_pid']}/stat").read_text().rsplit(')',1)[1].split()[0]
   if st=='Z':raise RuntimeError('Timing worker stopped before completion')
  except FileNotFoundError:raise RuntimeError('Timing worker absent before completion')
  save();time.sleep(15)
 for stage in plan['stages']:
  for path,digest in stage['pinned_sha256'].items():assert sha(ROOT/path)==digest,path
  status.update(state='running',stage=stage['name']);save()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
  with (BASE/(stage['name']+'.log')).open('xb') as log:
   p=subprocess.Popen([sys.executable,'-u',*stage['arguments']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   status['active_child_pid']=p.pid;save();code=p.wait()
  if code:raise RuntimeError(f"Stage {stage['name']} failed with exit{code}; no automatic retry")
  marker=json.loads((ROOT/stage['completion_file']).read_text())
  assert marker[stage['success_key']] is True,stage['name']
  status['completed_stages'].append(stage['name']);status.pop('active_child_pid',None);save()
 status.update(state='complete',next='Verify/render remaining figure4/8/15, review candidate gate without retuning, continue independent baseline/resource/audit gates. Overall thesis goal is not complete.');save()
except BaseException as error:
 status.update(state='stopped_requires_review',error=f'{type(error).__name__}: {error}');save();raise
