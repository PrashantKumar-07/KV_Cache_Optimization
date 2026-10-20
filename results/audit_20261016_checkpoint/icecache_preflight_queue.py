"""Follow the existing reviewed queue with baseline validation; never duplicate it."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((BASE/'icecache_preflight_queue_plan.json').read_text());assert not (BASE/'icecache_preflight_queue_status.json').exists()
state={'state':'waiting_for_registered_queue','pid':os.getpid(),'plan_sha256':sha(BASE/'icecache_preflight_queue_plan.json'),'completed_stages':[]}
def save():
 state['utc']=datetime.now(timezone.utc).isoformat();p=BASE/'icecache_preflight_queue_status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');os.replace(p,BASE/'icecache_preflight_queue_status.json')
try:
 while True:
  q=json.loads((BASE/'proper_comparison_status.json').read_text())
  if q['state']=='complete':break
  if q['state']=='stopped_requires_review':raise RuntimeError('Prior queue failed; inspect it instead of continuing')
  command=Path(f"/proc/{q['pid']}/cmdline").read_bytes();assert b'proper_comparison_queue.py' in command,'Prior queue not live'
  save();time.sleep(30)
 for stage in plan['stages']:
  for p,h in stage['sha256'].items():assert sha(ROOT/p)==h,p
  if stage.get('gpu'):
   uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
   assert not any(r.split(',')[0].strip()==uuid for r in apps),'GPU1 occupied; requires review before launch'
  state.update(state='running',stage=stage['name']);save()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
  env.update(stage.get('environment',{}))
  with (BASE/(stage['name']+'.log')).open('xb') as log:
   child=subprocess.Popen([stage.get('python',sys.executable),'-u',stage['script']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   state['child_pid']=child.pid;save();code=child.wait()
  if code:raise RuntimeError(f"{stage['name']} failed exit{code}; no automatic retry")
  marker=json.loads((ROOT/stage['output']).read_text());assert marker[stage['success']] is True
  state['completed_stages'].append(stage['name']);state.pop('child_pid',None);save()
 state.update(state='complete',next='Inspect compiledofficialkernel/tiny-model preflight proof. Fullpretrained7B/DCI-offload correctness andsame-runtime/resource-fair comparisons are STILL REQUIRED beforeclaims.');save()
except BaseException as error:
 state.update(state='stopped_requires_review',error=f'{type(error).__name__}: {error}');save();raise
