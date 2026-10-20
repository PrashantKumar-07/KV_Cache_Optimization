"""Wait for the two owned workers, then run CPU verification exactly once."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(value):
 value['utc']=datetime.now(timezone.utc).isoformat()
 temporary=BASE/'watch_status.tmp'
 temporary.write_text(json.dumps(value,indent=2)+'\n')
 os.replace(temporary,BASE/'watch_status.json')
def alive(pid):
 try:
  state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
  command=Path(f'/proc/{pid}/cmdline').read_bytes()
  return state!='Z' and b'run_full.py' in command
 except FileNotFoundError:return False
parser=argparse.ArgumentParser()
parser.add_argument('--launch',default='launch2.json')
args=parser.parse_args()
launch=json.loads((BASE/args.launch).read_text())
assert sha(BASE/'plan.json')==launch['plan_sha256']
verifier=BASE/'verify_results.py';verifier_hash=sha(verifier)
assert not (BASE/'verified_results.json').exists()
while True:
 complete=all((BASE/'records'/f'shard_{s}'/'complete.json').exists() for s in range(2))
 if complete:
  assert sha(verifier)==verifier_hash, 'Verifier changed after watcher registration'
  save({'state':'running_cpu_verification','verifier_sha256':verifier_hash})
  with (BASE/'final_verification.log').open('xb') as log:
   result=subprocess.run([sys.executable,str(verifier)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  save({'state':'verified' if result.returncode==0 else 'verification_failed','returncode':result.returncode,'verifier_sha256':verifier_hash,'results_final_only_if_verified':True})
  break
 living=[j['pid'] for j in launch['jobs'] if alive(j['pid'])]
 if not living:
  save({'state':'workers_stopped_before_complete','action':'Inspect preserved logs/errors; diagnose and resume exact frozen shards; no automatic favorable retry.'})
  break
 save({'state':'waiting_for_full_evaluation','live_owned_workers':living,'completed_request_files':sum(1 for _ in (BASE/'records').glob('shard_*/*_*.json')),'expected_request_files':10318,'verifier_sha256':verifier_hash})
 time.sleep(30)
