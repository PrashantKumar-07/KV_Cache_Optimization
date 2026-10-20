"""CPU proof of the preregistered long-prompt payload boundary, excluding metadata."""
from pathlib import Path
import hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from longbench_eval import TieredKVPolicy
from snapkv_adapter import upstream_module
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.manual_seed(0)
k=torch.randn(1,8,3000,4,dtype=torch.bfloat16);v=torch.randn_like(k)
p=TieredKVPolicy(sram_budget=1024,stt_budget=2048,stt_expose_quota=32,
    start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
p.prompt_q_by_layer={0:torch.randn(32,128,4,dtype=torch.bfloat16)}
visible=p(((k,v),));cache=p.layer_caches[0]
tensors=[cache.sram_k,cache.sram_v,cache.stt_k,cache.stt_v,*visible[0]]
storages={x.untyped_storage()._cdata:x.untyped_storage().nbytes() for x in tensors}
tiered_bytes=sum(storages.values())
reference=upstream_module();cluster=reference.SnapKVCluster(32,904,5,'avgpool')
k32=reference.repeat_kv(k,4);v32=reference.repeat_kv(v,4)
q=torch.randn(1,32,3000,4,dtype=torch.bfloat16)
snap=cluster.update_kv(k32,q,v32,None,4)
snap_bytes=sum({x.untyped_storage()._cdata:x.untyped_storage().nbytes() for x in snap}.values())
assert cache.sram_k.shape[1]==1024 and cache.stt_k.shape[1]==1536
assert visible[0][0].shape[2]==1056 and snap[0].shape[2]==904
assert tiered_bytes==snap_bytes==462848
report={'passed':True,'tiered_unique_retained_payload_bytes_one_layer_D4':tiered_bytes,
    'snap_unique_retained_payload_bytes_one_layer_D4':snap_bytes,
    'tiered_tensor_shapes':[list(x.shape) for x in tensors],'snap_tensor_shapes':[list(x.shape) for x in snap],
    'boundary':'Long-prompt initial held KV tensor storage only, per-layer D4 geometry proxy. Excludes metadata, transient/full prefill state/modelweights, framework allocations; equality does not persist through growing decode or shorter prompts.',
    'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['experiments/longbench_eval.py','src/tiered_kv_cache.py','experiments/snapkv_adapter.py','sota/snapkv/snapkv/monkeypatch/snapkv_utils.py']}}
output=BASE/'verified_prompt_payload.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
