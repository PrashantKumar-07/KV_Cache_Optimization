"""Frozen current native-controller performance/resource gate; guarded ownership."""
from contextlib import ExitStack
from pathlib import Path
import gc,hashlib,importlib.util,json,os,sys
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from exact_cold_boundary_policy import ExactColdBoundaryPolicy
from exact_pinned_memory import stats as allocation_stats
from longbench_eval import TieredKVPolicy
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from latency_profile import profile_one
from audit_pinned_blocks import kernel as pinned_probe
from provenance import atomic_json,model_artifact_identity,runtime_identity
plan=json.loads((BASE/'plan.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (BASE/'timing.json').exists()
prerequisite=ROOT/plan['parity_prerequisite']
proof=json.loads(prerequisite.read_text())
assert proof['verified'] and proof['finite_and_bytes_exact']
assert all(sha(ROOT/n)==v for n,v in plan['source_sha256'].items())
for key in ('prompt','teacher_stream','quiet_helper'):
 assert sha(ROOT/plan[key])==plan[key+'_sha256']
spec=importlib.util.spec_from_file_location('quiet_observer',ROOT/plan['quiet_helper'])
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
report={'passed':False,'rows':[],'quiet_observations':[],'plan_sha256':sha(BASE/'plan.json'),
    'runtime':runtime_identity(),'pid':os.getpid(),'parity_prerequisite_sha256':sha(prerequisite),'model_identity':model_artifact_identity('../models/mistral-7b-instruct')}
def quiet(stage):
 observation=dict(stage=stage,**observer.quiet());report['quiet_observations'].append(observation)
 atomic_json(BASE/'timing.json',report);assert observation['quiet'],'Unexpected GPU0 compute process'
try:
 quiet('before_model_load');torch.manual_seed(0)
 model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,
     device_map='cuda',attn_implementation='eager').eval()
 tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
 static={t.untyped_storage().data_ptr():t.untyped_storage().nbytes()
         for t in [*model.parameters(),*model.buffers()] if t.device.type=='cuda'}
 report['constant_model_storage_bytes']=sum(static.values())
 stream=json.loads((ROOT/plan['teacher_stream']).read_text())['config']['teacher_token_ids']
 for index,arm in enumerate(plan['order']):
  quiet('before_row')
  before=(len(model._forward_pre_hooks),len(model._forward_hooks),
      [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers])
  with ExitStack() as cleanup:
   policy=(TieredKVPolicy if arm=='control' else ExactColdBoundaryPolicy)(sram_budget=1024,stt_budget=2048,
       stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
   cleanup.callback(policy.reset)
   capture=PromptQueryCapture(model,policy);cleanup.callback(capture.close)
   lease=None
   if arm=='candidate':
    lease=SharedAppendLease(model,policy);cleanup.callback(lease.close)
   result=profile_one(model,tokenizer,policy,16384,n_decode=50,n_warmup=1,n_repeat=2,
       device='cuda',prompt_text=(ROOT/plan['prompt']).read_text(),forced_token_ids=stream,prefill_last_logits=True)
   row={'index':index,'arm':arm,'result':result}
   if arm=='candidate':
    fields=['cold_boundary_steps','cold_boundary_fallback_calls','cpu_append_calls','cpu_fallback_steps',
        'bulk_payload_steps','bulk_payload_fallback_steps','bulk_descriptor_h2d_bytes',
        'cold_boundary_descriptor_h2d_bytes','controller_cpu_streak_array_peak','controller_address_array_peak']
    row['counts']={name:getattr(policy,name,0) for name in fields}
    row['counts']['controller_address_array_peak']=getattr(policy.manager,'controller_address_array_peak',0)
    row['owned_pinned_pool']=pinned_probe().block_info(policy.manager.pool)
    row['cpu_canonical_arrays_bytes']=policy.cpu_arena.owned_bytes
   report['rows'].append(row)
  row['hooks_closed']=before==(len(model._forward_pre_hooks),len(model._forward_hooks),
      [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers])
  row['lease_closed']=lease is None or (not lease.handles and not lease.pending)
  assert row['hooks_closed'] and row['lease_closed']
  del policy,capture,lease;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
  row['exact_allocation_stats_after_cleanup']=allocation_stats()
  assert allocation_stats()['live_allocations']==0
  quiet('after_row');atomic_json(BASE/'timing.json',report)
  print(index,arm,result['itl_ms'],'ms/token',flush=True)
 report['passed']=True
finally:
 atomic_json(BASE/'timing.json',report)
