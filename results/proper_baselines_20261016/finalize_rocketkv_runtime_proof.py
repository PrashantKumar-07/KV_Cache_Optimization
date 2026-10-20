"""Durable isolated official RocketKV runtime installation; zero GPU access."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,traceback
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];SOURCE=BASE/'RocketKV';PY=BASE/'venv_rocketkv/bin/python'
PROTOCOL=BASE/'rocketkv_runtime_protocol.json'
state={'state':'starting','pid':os.getpid(),'stages_complete':[]}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MAX_JOBS='4',CMAKE_BUILD_PARALLEL_LEVEL='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUDA_HOME='/usr/local/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.9',PIP_DISABLE_PIP_VERSION_CHECK='1',TMPDIR=str(BASE/'rocketkv_tmp'))
def save():
 state['utc']=datetime.now(timezone.utc).isoformat();tmp=BASE/'rocketkv_runtime_final_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,BASE/'rocketkv_runtime_final_status.json')
def run(name,args):
 state.update(state='installing',stage=name,command=[str(x) for x in args]);save()
 with (BASE/(name+'_final.log')).open('xb') as log:
  p=subprocess.Popen([str(x) for x in args],cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
  state['child_pid']=p.pid;save();rc=p.wait()
 state.pop('child_pid',None)
 if rc:raise RuntimeError(f'{name} failed exit{rc}; preserved log must be inspected before any retry')
 state['stages_complete'].append(name);save()
try:
 protocol=json.loads(PROTOCOL.read_text());assert PY.exists()
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==protocol['source_commit']
 for name,digest in protocol['source_sha256'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
 import urllib.request
 wheels=BASE/'rocketkv_official_wheels';wheels.mkdir(exist_ok=True)
 wheel=wheels/'flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
 state.update(state='verifying_wheel_provenance',stage='download_official_wheel_for_hash');save()
 assert not wheel.exists()
 with urllib.request.urlopen(protocol['flash_attention_wheel_url'],timeout=90) as src, wheel.open('xb') as dst:
  while True:
   block=src.read(1024*1024)
   if not block:break
   dst.write(block)
 state['official_wheel_sha256']=hashlib.sha256(wheel.read_bytes()).hexdigest();state['stages_complete'].append('official_wheel_downloaded_for_installed_byte_comparison');save()
 run('rocketkv_cpu_import_contract',[PY,BASE/'check_rocketkv_cpu.py'])
 run('rocketkv_pip_check',[PY,'-m','pip','check'])
 with (BASE/'rocketkv_pip_freeze.txt').open('w') as out:subprocess.run([str(PY),'-m','pip','freeze'],env=env,stdout=out,check=True)
 state.update(state='cpu_import_and_model_construction_validated',patch_installation_verified=False,gpu_forward_verified=False,quality_or_latency_claim=False,protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest());save()
except BaseException as error:
 state.update(state='stopped_requires_review',error=f'{type(error).__name__}: {error}');save();traceback.print_exc();raise
