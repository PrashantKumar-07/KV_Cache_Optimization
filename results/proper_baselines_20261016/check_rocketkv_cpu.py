"""CPU-only import/patch installation contract; no inference correctness claim."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,importlib.metadata as metadata
BASE=Path(__file__).resolve().parent; SOURCE=BASE/'RocketKV'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch,transformers,flash_attn,accelerate,numpy
assert not torch.cuda.is_initialized()
assert torch.__version__=='2.4.1+cu124'
assert transformers.__version__=='4.45.2'
assert flash_attn.__version__=='2.6.3'
assert accelerate.__version__=='0.34.2' and numpy.__version__=='1.26.4'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()=='63637c8eb4b06b5eb83cd687723b0eeb19f48f65'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=SOURCE,text=True).strip()
sys.path.insert(0,str(SOURCE/'pipeline/inf_stream_llm'))
from inf_llm import patch_hf
from transformers import MistralConfig,MistralForCausalLM
config=MistralConfig(vocab_size=257,hidden_size=128,intermediate_size=256,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,sliding_window=None,rope_theta=1000000.0)
config._attn_implementation='eager'
model=MistralForCausalLM(config).eval()
from inf_llm.attention.rocket import rocket_forward
import ast,inspect
forward=rocket_forward(fattn=True,topk=64,compression_ratio=4.0,prompt_budget=256,window_size=32,kernel_size=63,skip_layers=0)
assert callable(forward) and callable(patch_hf)
assert all(p.device.type=='cpu' for p in model.parameters())
from inf_llm.attention.rope import RotaryEmbeddingESM
rope_ast=ast.parse(inspect.getsource(RotaryEmbeddingESM))
cuda_allocations=[n for n in ast.walk(rope_ast) if isinstance(n,ast.Call) and any(k.arg=='device' and isinstance(k.value,ast.Constant) and k.value.value=='cuda' for k in n.keywords)]
assert len(cuda_allocations)==1
assert not torch.cuda.is_initialized()
direct_url=json.loads(metadata.distribution('flash-attn').read_text('direct_url.json'))
assert direct_url['url'].startswith('https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/')
import zipfile
wheel=BASE/'rocketkv_official_wheels/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
wheel_digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
verified_wheel_members=0
with zipfile.ZipFile(wheel) as archive:
 for member in archive.namelist():
  if member.endswith(('.py','.so')) and '.dist-info/' not in member:
   installed=Path(metadata.distribution('flash-attn').locate_file(member))
   assert hashlib.sha256(installed.read_bytes()).digest()==hashlib.sha256(archive.read(member)).digest(),member
   verified_wheel_members+=1
assert verified_wheel_members>10
modules={x:metadata.version(x) for x in ('torch','transformers','flash-attn','accelerate','numpy','huggingface-hub','sentencepiece','tokenizers','safetensors')}
module_hashes={}
for name in ('inf_llm/utils/patch.py','inf_llm/attention/rocket.py','inf_llm/attention/rope.py'):
 p=SOURCE/'pipeline/inf_stream_llm'/name
 module_hashes[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
report={'cpu_import_and_model_construction_verified':True,'official_forward_factory_constructed':True,'patch_hf_installation_verified':False,'patch_hf_deferred_reason':'Official RotaryEmbeddingESM.__init__ unconditionally allocates inv_freq on CUDA; no device mocks or upstream edits used. Real installation plus kernel/forward correctness requires queued GPU validation.','gpu_kernel_or_forward_verified':False,'quality_or_latency_claim':False,'cuda_initialized':torch.cuda.is_initialized(),'model':'random two-layer Mistral on CPU, no forward call','modules':modules,'torch_cuda_build':torch.version.cuda,'torch_cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,'source_commit':'63637c8eb4b06b5eb83cd687723b0eeb19f48f65','source_hashes':module_hashes,'flash_attention_official_wheel_direct_url':direct_url,'flash_attention_official_wheel_sha256':wheel_digest,'installed_wheel_module_files_verified':verified_wheel_members,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(BASE/'rocketkv_cpu_import_proof.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
