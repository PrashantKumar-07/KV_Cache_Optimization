"""Resource trajectories only: native controller vs faithful Snap selector.

Copies and synchronization deliberately invalidate timing from this diagnostic.
No total-memory equality or quality claim follows from these fixed capacities.
"""
from contextlib import ExitStack
from pathlib import Path
import gc, hashlib, json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parents[3]; BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from reusable_exact_root import ExactRootOwner, ReusableExactPolicy
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from latency_profile import profile_one
from longbench_eval import _attn_impl, prefill_logit_kwargs
from snapkv_adapter import SnapKVAdapter
from audit_snap_tier_resources import distinct_storage_bytes
from audit_pinned_blocks import kernel as pinned_probe
from exact_pinned_memory import stats
from provenance import atomic_json, model_artifact_identity, runtime_identity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((BASE/'plan.json').read_text())
assert not (BASE/'diagnostic.json').exists()
assert sha(Path(__file__))==plan['runner_sha256']
assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
assert model_artifact_identity(plan['model'])==plan['model_identity']
for name in ['prompt','teacher_stream']:assert sha(ROOT/plan[name])==plan[name+'_sha256']
owner=ExactRootOwner(); report={'passed':False,'plan_sha256':sha(BASE/'plan.json'),'runtime':runtime_identity(),'arms':{}}
def boundary(visible,policy=None):
 tensors=[t for pair in visible for t in pair]; layers=[]
 for i,(k,v) in enumerate(visible):
  row={'layer':i,'visible_rows':k.shape[-2],'visible_heads':k.shape[-3]}
  if policy is not None:
   c=policy.layer_caches[i];tensors.extend([c.sram_k,c.sram_v,c.pool])
   row.update(hot_rows=len(c.sram_pos),cold_total_rows=len(c.stt_pos),cold_live_rows=int((~c.stt_shadow).sum().item()))
  layers.append(row)
 return {'layers':layers,'deduplicated_named_KV_storage_bytes':distinct_storage_bytes(tensors),
  'visible_tensor_payload_bytes':sum(t.numel()*t.element_size() for pair in visible for t in pair),
  'gpu_allocated_bytes':torch.cuda.memory_allocated(),'gpu_reserved_bytes':torch.cuda.memory_reserved(),
  'cpu_canonical_metadata_bytes':0 if policy is None else policy.cpu_arena.owned_bytes}
class Candidate(ReusableExactPolicy):
 def __init__(self,**kw):super().__init__(**kw);self.trace=[]
 def _bifurcate(self,*args,**kw):out=super()._bifurcate(*args,**kw);self.trace.append(boundary(out,self));return out
 def _decode_step(self,*args,**kw):out=super()._decode_step(*args,**kw);self.trace.append(boundary(out,self));return out
def hooks(model):return ([len(l.self_attn._forward_pre_hooks) for l in model.model.layers],len(model._forward_pre_hooks),len(model._forward_hooks),[l.self_attn.num_key_value_groups for l in model.model.layers])
try:
 torch.manual_seed(0)
 with torch.inference_mode():
  model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
  tokenizer=AutoTokenizer.from_pretrained(plan['model']);prompt=(ROOT/plan['prompt']).read_text()
  ids=tokenizer(prompt,return_tensors='pt').input_ids
  assert ids.shape[1]>=16384;ids=ids[:,:16384].to('cuda')
  stream=json.loads((ROOT/plan['teacher_stream']).read_text())['config']['teacher_token_ids'][:51]
  assert len(stream)==51;report['input_ids_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
  static={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in [*model.parameters(),*model.buffers()]}
  report['constant_model_storage_bytes']=sum(static.values())
  before=hooks(model);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  with ExitStack() as cleanup:
   cache=SnapKVAdapter(model,capacity=904,window=32);cleanup.callback(cache.close)
   with _attn_impl(model,'sdpa'):
    out=model(ids,past_key_values=cache,use_cache=True,position_ids=torch.arange(16384,device='cuda').unsqueeze(0),**prefill_logit_kwargs(model,True))
   del out
   trace=[boundary([(l.keys,l.values) for l in cache.layers])];pred=[]
   for step in range(50):
    out=model(torch.tensor([[stream[step]]],device='cuda'),past_key_values=cache,use_cache=True,position_ids=torch.tensor([[16384+step]],device='cuda'))
    assert torch.isfinite(out.logits).all();pred.append(out.logits[:,-1].argmax(-1).item());del out
    trace.append(boundary([(l.keys,l.values) for l in cache.layers]))
   report['arms']['snap']={'boundaries':trace,'predicted_token_ids':pred,'diagnostic_peak_gpu_allocated':torch.cuda.max_memory_allocated()}
  assert hooks(model)==before
  del cache;gc.collect();torch.cuda.empty_cache();atomic_json(BASE/'diagnostic.json',report)
  policy=Candidate(root_owner=owner,sram_budget=1024,stt_budget=2048,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
  with ExitStack() as cleanup:
   cleanup.callback(policy.reset)
   capture=PromptQueryCapture(model,policy);cleanup.callback(capture.close)
   lease=SharedAppendLease(model,policy);cleanup.callback(lease.close)
   result=profile_one(model,tokenizer,policy,16384,n_decode=50,n_warmup=0,n_repeat=1,device='cuda',prompt_text=prompt,forced_token_ids=stream,prefill_last_logits=True)
   report['arms']['native']={'boundaries':policy.trace,'predicted_token_ids':result['trials'][0]['predicted_token_ids'],
    'actual_pinned_pool':pinned_probe().block_info(owner.root),
    'counts':{n:getattr(policy,n,0) for n in ['cold_boundary_steps','cpu_append_calls','cpu_fallback_steps','cold_boundary_fallback_calls']}}
  assert hooks(model)==before and owner.token is None
  report['retained_idle_pool']=pinned_probe().block_info(owner.root)
  del policy,capture,lease,result;gc.collect();owner.close();gc.collect();torch.cuda.synchronize()
  report['final_allocation_stats']=stats();assert stats()['live_allocations']==0
  assert all(len(a['boundaries'])==51 for a in report['arms'].values())
  report['passed']=True
finally:
 if owner.token is None and not owner.closed:owner.close()
 report['allocation_stats_at_exit']=stats();atomic_json(BASE/'diagnostic.json',report)
