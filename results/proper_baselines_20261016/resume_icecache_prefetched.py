"""Build official BF16 CPU-offload baseline without touching the thesis runtime."""
from pathlib import Path
import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;PY=BASE/'venv_icecache/bin/python'
state={'state':'resuming_after_verified_compiler_diagnosis','pid':os.getpid(),'stages_complete':[]}
def save():
 state['utc']=datetime.now(timezone.utc).isoformat();p=BASE/'build_resume5_status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');os.replace(p,BASE/'build_resume5_status.json')
def run(name,args,cwd=BASE):
 state.update(state='building',stage=name);save()
 with (BASE/(name+'.log')).open('xb') as log:
  p=subprocess.Popen([str(PY),*args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
  state['child_pid']=p.pid;save();rc=p.wait()
 if rc:raise RuntimeError(f'{name} failed exit{rc}; inspect preserved logs before correcting build')
 state['stages_complete'].append(name);state.pop('child_pid',None);save()
lib=BASE/'local_openblas/usr/lib/x86_64-linux-gnu/openblas-pthread';include=BASE/'local_openblas/usr/include/x86_64-linux-gnu/openblas-pthread'
env=dict(os.environ,CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',CUDACXX='/usr/local/cuda-12.4/bin/nvcc',CUDAHOSTCXX='/usr/bin/g++-11',MAX_JOBS='8',CMAKE_BUILD_PARALLEL_LEVEL='8',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',CUDA_HOME='/usr/local/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.9',CPATH=str(include),LIBRARY_PATH=str(lib),LD_LIBRARY_PATH=str(lib)+':'+str(BASE/'venv_icecache/lib/python3.10/site-packages/torch/lib'),PATH=str(BASE/'venv_icecache/bin')+':/usr/local/cuda-12.4/bin:'+os.environ['PATH'])
try:
 for _ in range(120):
  result=subprocess.run([str(PY),'-c','import torch;assert torch.__version__.startswith("2.4.0")'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  if result.returncode==0:break
  if (BASE/'torch_install.log').exists() and 'ERROR:' in (BASE/'torch_install.log').read_text():raise RuntimeError('Torch isolated installation failed')
  save();time.sleep(15)
 else:raise RuntimeError('Torch installation not available after30min; check original installer')
 run('build_official_icecache_prefetched',['-m','pip','install','-e','.','--no-build-isolation'],BASE/'IceCache-bf16/IceCache/source')
 run('cpu_import_contract',['-c','import torch,transformers,flash_attn,dci,icecache;assert hasattr(dci,"PyDCI");assert all(hasattr(dci,x) for x in ("reuse_copy_node","reuse_update_node","copy_to_buffer","diff_pages_by_head"));print(torch.__version__,transformers.__version__);print("Official module import contracts passed; GPU validation still required")'])
 with (BASE/'pip_freeze.txt').open('w') as out:subprocess.run([str(PY),'-m','pip','freeze'],env=env,stdout=out,check=True)
 state.update(state='built_import_validated',gpu_validation=False,quality_or_latency_claim=False,next='Run upstream kernel/model correctness checks onGPU after currentGQAquality run completes; preregister matchedresources/protocol and controls in this same runtime. Do not compare cross-environment walltimes as matched evidence.');save()
except BaseException as error:
 state.update(state='stopped_requires_build_review',error=f'{type(error).__name__}: {error}');save();raise
