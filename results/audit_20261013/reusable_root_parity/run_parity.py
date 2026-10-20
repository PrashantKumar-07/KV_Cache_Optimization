"""Preregistered sequential-request root reuse parity/lifetime diagnostic."""
from contextlib import ExitStack
from pathlib import Path
import gc,hashlib,json,sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from longbench_eval import TieredKVPolicy
from reusable_exact_root import ExactRootOwner,ReusableExactPolicy
from exact_pinned_memory import stats
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from latency_profile import profile_one
from audit_pinned_blocks import kernel as pinned_probe
from provenance import atomic_json,model_artifact_identity,runtime_identity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((BASE/'plan.json').read_text());assert not (BASE/'parity.json').exists()
assert sha(BASE/'run_parity.py')==plan['runner_sha256']
assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
for k in ['prompt','teacher_stream']:assert sha(ROOT/plan[k])==plan[k+'_sha256']
identity=model_artifact_identity(plan['model']);assert identity==plan['model_identity']
stream=json.loads((ROOT/plan['teacher_stream']).read_text())['config']['teacher_token_ids']
report={'passed':False,'pairs':{},'plan_sha256':sha(BASE/'plan.json'),'runtime':runtime_identity(),'model_identity':identity}
owner=ExactRootOwner()
def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
class Record:
 def __init__(self,**options):super().__init__(**options);self.trace=[]
 def snapshot(self,visible):
  torch.cuda.synchronize();rows={}
  for i,c in self.layer_caches.items():
   rows[str(i)]={'hot_pos':list(c.sram_pos),'cold_pos':list(c.stt_pos),'cold_shadow':c.stt_shadow.tolist(),
    'hot_k':tensor_hash(c.sram_k),'hot_v':tensor_hash(c.sram_v),'cold_k':tensor_hash(c.stt_k),'cold_v':tensor_hash(c.stt_v),
    'visible_k':tensor_hash(visible[i][0]),'visible_v':tensor_hash(visible[i][1]),
    'hot_cumulative':tensor_hash(c.sram_cum),'hot_age':tensor_hash(c.sram_age),'cold_last_access':tensor_hash(c.stt_last_access),
    'cold_peek_streak':tensor_hash(c.stt_peek_streak),'cold_was_hot':tensor_hash(c.stt_was_vram_resident)}
  self.trace.append(rows)
 def _bifurcate(self,*args,**kw):out=super()._bifurcate(*args,**kw);self.snapshot(out);return out
 def _decode_step(self,*args,**kw):out=super()._decode_step(*args,**kw);self.snapshot(out);return out
class Control(Record,TieredKVPolicy):pass
class Candidate(Record,ReusableExactPolicy):pass
options=dict(sram_budget=1024,stt_budget=2048,stt_expose_quota=32,start_size=4,recent_size=128,
 kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
try:
 torch.manual_seed(0)
 model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
 assert model.config.model_type=='mistral' and model.config.sliding_window is None
 assert model.config.num_hidden_layers==32 and model.config.num_key_value_heads==8 and model.config.hidden_size//model.config.num_attention_heads==128
 tokenizer=AutoTokenizer.from_pretrained(plan['model']);prompt=(ROOT/plan['prompt']).read_text();all_ids=tokenizer(prompt,return_tensors='pt')['input_ids']
 pointer=None
 for context in plan['contexts']:
  assert all_ids.shape[1]>=context
  pair={};report['pairs'][str(context)]=pair
  report.setdefault('request_input_ids_sha256',{})[str(context)]=sha_ids=hashlib.sha256(all_ids[:,:context].numpy().tobytes()).hexdigest()
  for arm,kind in [('control',Control),('candidate',Candidate)]:
   before=(len(model._forward_pre_hooks),len(model._forward_hooks),[len(l.self_attn._forward_pre_hooks) for l in model.model.layers])
   policy=kind(**options,**({'root_owner':owner} if arm=='candidate' else {}));logits=[]
   with ExitStack() as cleanup:
    cleanup.callback(policy.reset)
    capture=PromptQueryCapture(model,policy);cleanup.callback(capture.close)
    lease=SharedAppendLease(model,policy) if arm=='candidate' else None
    if lease:cleanup.callback(lease.close)
    handle=model.register_forward_hook(lambda m,args,out:logits.append(out.logits.detach().float().cpu().numpy()));cleanup.callback(handle.remove)
    result=profile_one(model,tokenizer,policy,context,n_decode=50,n_warmup=0,n_repeat=1,device='cuda',prompt_text=prompt,forced_token_ids=stream,prefill_last_logits=True)
    np.savez_compressed(BASE/f'{context}_{arm}_logits.npz',**{str(i):v for i,v in enumerate(logits)})
    row={'trace':policy.trace,'predicted_token_ids':result['trials'][0]['predicted_token_ids'],'forwards':len(logits)}
    if arm=='candidate':
     row['counts']={n:getattr(policy,n,0) for n in ['cold_boundary_steps','cold_boundary_fallback_calls','cpu_append_calls','cpu_fallback_steps','bulk_payload_steps','bulk_payload_fallback_steps']}
     row['pool_pointer']=policy.manager.pool.data_ptr();row['pool_info']=pinned_probe().block_info(policy.manager.pool)
     assert policy.manager.pool.data_ptr()==owner.root.data_ptr()
     if pointer is None:pointer=row['pool_pointer']
     assert pointer==row['pool_pointer']
   row['hooks_closed']=before==(len(model._forward_pre_hooks),len(model._forward_hooks),[len(l.self_attn._forward_pre_hooks) for l in model.model.layers])
   row['lease_closed']=lease is None or not lease.handles and not lease.pending
   del policy,capture,lease;gc.collect();torch.cuda.synchronize()
   row['allocation_stats_after_request_reset']=stats();row['owner_idle']=owner.token is None
   if arm=='candidate':
    assert row['owner_idle'] and stats()['live_allocations']==1 and stats()['live_requested_bytes']==276824064
    row['idle_pool_info']=pinned_probe().block_info(owner.root)
   pair[arm]=row;atomic_json(BASE/'parity.json',report)
   print(context,arm,'forwards',len(logits),flush=True)
  left,right=pair['control'],pair['candidate'];assert left['trace']==right['trace'] and left['predicted_token_ids']==right['predicted_token_ids']
  assert all(r['forwards']==51 and len(r['trace'])==51 and r['hooks_closed'] and r['lease_closed'] for r in pair.values())
  with np.load(BASE/f'{context}_control_logits.npz') as a,np.load(BASE/f'{context}_candidate_logits.npz') as b:
   assert set(a.files)==set(b.files)=={str(i) for i in range(51)}
   assert all(np.isfinite(a[str(i)]).all() and np.isfinite(b[str(i)]).all() and a[str(i)].tobytes()==b[str(i)].tobytes() for i in range(51))
  assert right['counts']['cold_boundary_steps']==right['counts']['cpu_append_calls']==50 and right['counts']['cpu_fallback_steps']==0
  assert right['counts']['cold_boundary_fallback_calls']==1
 # Failure after directfactory attachment:32layergeometry matches retained root.
 probe=ReusableExactPolicy(root_owner=owner,**options)
 tiny=torch.zeros((1,8,16,128),dtype=torch.bfloat16,device='cuda')
 probe.prompt_q_by_layer={0:torch.empty((0,16,128),dtype=torch.bfloat16,device='cuda')}
 try:probe(tuple((tiny,tiny) for _ in range(32)))
 except ValueError as error:
  assert 'Captured prompt query shape' in str(error)
  report['lifecycle_failure_probe']={'error':str(error),'registered_layer_caches':len(probe.layer_caches),'attached_manager_caches':len(probe.manager.caches),'owner_busy_before_reset':owner.token is not None}
 else:raise AssertionError('Malformed capturedquery did not reject')
 attached=list(probe.manager.caches.values());assert len(attached)==1 and not probe.layer_caches and owner.token is not None
 probe.reset()
 assert all(c.pool is None and c.manager is None for c in attached) and owner.token is None and stats()['live_allocations']==1
 report['lifecycle_failure_probe'].update(attached_views_closed=True,owner_idle_after_reset=True,retained_root_pointer=owner.root.data_ptr(),allocation_stats_after_reset=stats())
 assert owner.root.data_ptr()==pointer and owner.leases==3
 del probe,attached,tiny;gc.collect()
 report['retained_idle_pool_info']=pinned_probe().block_info(owner.root)
 assert report['retained_idle_pool_info']['allocator_block_bytes']==report['retained_idle_pool_info']['requested_storage_bytes']==276824064
 report['owner_allocation_ms_observation']=owner.allocation_ms # Diagnostic provisioning only, not servingTTFT.
 report['owner_leases_including_failure_probe']=owner.leases
 owner.close();gc.collect();torch.cuda.synchronize();report['allocation_stats_after_final_close']=stats()
 assert stats()['live_allocations']==stats()['live_requested_bytes']==0
 events=stats()['events'];assert len(events)==2 and events[0]['kind']=='allocate' and events[1]['kind']=='release' and events[0]['token']==events[1]['token'] and events[0]['bytes']==events[1]['bytes']==276824064
 report['passed']=True
finally:
 if not owner.closed and owner.token is None:owner.close();gc.collect()
 report['allocation_stats_at_exit']=stats();atomic_json(BASE/'parity.json',report)
