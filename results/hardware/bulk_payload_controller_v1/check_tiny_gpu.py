"""Real multi-token fallback and return to canonical CPU native state."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import torch
from transformers import MistralConfig,MistralForCausalLM
from bulk_promotion_policy import BulkPromotionPolicy
from cpu_phase_policy import CpuPhasePolicy
from early_shared_cpu_phase import EarlySharedPolicy,SharedPrefillLease
from shared_decode_transition import SharedAppendLease
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import _attn_impl,apply_tuple_kv_inplace,to_tuple_kv,prefill_logit_kwargs
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bits(value):
 value=value.detach().cpu().contiguous()
 return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
def frame(policy,past,logits):
 return dict(state_sha256=digest(policy.layer_caches),counters=policy._live_totals(),logits=bits(logits),predicted_tokens=logits.argmax(-1).cpu().tolist(),visible=[[bits(k),bits(v)] for k,v in to_tuple_kv(past)],layers=[dict(hot_k=bits(c.sram_k),hot_v=bits(c.sram_v),cold_k=bits(c.stt_k),cold_v=bits(c.stt_v),logical=list(c._logical),free=list(c._free),grace=bits(c.sram_grace),resident=bits(c.stt_was_vram_resident),streak=bits(c.stt_peek_streak),time=c._t) for c in policy.layer_caches.values()])
@torch.inference_mode()
def run(model,cls):
 p=cls(sram_budget=191,stt_budget=256,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
 p.prompt_query_score_mode='chunked';p.prompt_query_chunk=128
 before=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
 prefill_lease=SharedPrefillLease(model,p) if cls is EarlySharedPolicy else None
 lease=SharedAppendLease(model,p) if issubclass(cls,CpuPhasePolicy) else None
 capture=PromptQueryCapture(model,p);past=out=None;frames=[];counts=[]
 try:
  ids=(torch.arange(512,device='cuda')%127)[None]
  with _attn_impl(model,'sdpa'):out=model(ids,use_cache=True,position_ids=torch.arange(512,device='cuda')[None],**prefill_logit_kwargs(model,True))
  past=out.past_key_values;apply_tuple_kv_inplace(past,p(to_tuple_kv(past)));capture.close();capture=None
  frames.append(frame(p,past,out.logits[:,-1:]));del out
  pos=512
  for n in [1,2,1,32,1,1]:
   tokens=(torch.arange(pos,pos+n,device='cuda')%127)[None]
   out=model(tokens,past_key_values=past,use_cache=True,position_ids=torch.arange(pos,pos+n,device='cuda')[None],output_attentions=True)
   past=out.past_key_values;tup=to_tuple_kv(past);p.update_scores(tup,out.attentions);apply_tuple_kv_inplace(past,p(tup))
   frames.append(frame(p,past,out.logits));del out;pos+=n
   if issubclass(cls,CpuPhasePolicy):counts.append(dict(native=p.cpu_demote_calls,fallback=p.cpu_fallback_steps,attached=p.cpu_arena is not None))
  return dict(frames=frames,dispatch=counts,early_shared_layers=getattr(p,'shared_prefill_layers',0),prototype_counts={n:sum(getattr(c,n) for c in p.layer_caches.values()) for n in ['host_read_bytes','host_write_bytes','host_initial_write_bytes']},native_h2d_calls=p.manager.h2d_calls,native_d2h_calls=p.manager.d2h_calls,bulk_steps=getattr(p,'bulk_payload_steps',0),bulk_fallback_steps=getattr(p,'bulk_payload_fallback_steps',0),bulk_dispatches=getattr(p.manager,'bulk_dispatches',0),append_fast_updates=lease.fast_updates if lease else 0,append_fallback_updates=lease.fallback_updates if lease else 0)
 finally:
  if capture:capture.close()
  if lease:lease.close()
  if prefill_lease:prefill_lease.close()
  p.reset();past=out=None
  after=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
  if before!=after:
   (BASE/(cls.__name__+"_cleanup_mismatch.json")).write_text(json.dumps(dict(before=before,after=after,active_hooks=[(n,len(m._forward_pre_hooks),len(m._forward_hooks)) for n,m in model.named_modules() if m._forward_pre_hooks or m._forward_hooks]),indent=2)+"\n")
  assert before==after
  if lease:assert not lease.handles and not lease.pending


from bulk_payload_policy import BulkPayloadPolicy
@torch.inference_mode()
def check_partial_queue_failure(model):
 from measure_fresh_process_memory import PinnedRoots
 import gc
 tracker=PinnedRoots();original_empty=torch.empty;torch.empty=tracker.empty
 p=BulkPayloadPolicy(sram_budget=191,stt_budget=256,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
 p.prompt_query_score_mode='chunked';p.prompt_query_chunk=128
 before=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
 lease=SharedAppendLease(model,p);capture=PromptQueryCapture(model,p);past=out=None;witness={};manager=None
 try:
  ids=(torch.arange(512,device='cuda')%127)[None]
  with _attn_impl(model,'sdpa'):out=model(ids,use_cache=True,position_ids=torch.arange(512,device='cuda')[None],**prefill_logit_kwargs(model,True))
  past=out.past_key_values;apply_tuple_kv_inplace(past,p(to_tuple_kv(past)));capture.close();capture=None;del out
  original=p._promotion_plan
  def fail(layer,cache):
   if layer==1:
    witness['queued_before_failure']=p.manager.bulk_pending is not None;witness['bulk_steps']=p.bulk_payload_steps
    raise RuntimeError('deliberate queued bulk phase failure')
   return original(layer,cache)
  p._promotion_plan=fail
  out=model(torch.tensor([[17]],device='cuda'),past_key_values=past,use_cache=True,position_ids=torch.tensor([[512]],device='cuda'),output_attentions=True)
  tup=to_tuple_kv(out.past_key_values);p.update_scores(tup,out.attentions);manager=p.manager
  try:p(tup)
  except RuntimeError as e:assert str(e)=='deliberate queued bulk phase failure'
  else:raise AssertionError('Expected queued failure')
  assert witness['queued_before_failure'] and witness['bulk_steps']==1
 finally:
  if capture:capture.close()
  lease.close();p.reset();past=out=None;torch.empty=original_empty
 gc.collect();torch.cuda.synchronize()
 assert tracker.current==0 and manager.bulk_pending is None and manager.pool is None
 assert p.manager is None and not p.layer_caches and not lease.handles and not lease.pending
 assert before==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
 return dict(**witness,pinned_after_cleanup_bytes=tracker.current,pinned_peak_bytes=tracker.peak,bulk_dispatches_after_cleanup=manager.bulk_dispatches,hooks_leases_policy_released=True)

def main():
 out=BASE/'gpu.json';assert not out.exists();r=dict(passed=False,plan_sha256=sha(BASE/'plan.json'))
 try:
  torch.manual_seed(7);config=MistralConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,vocab_size=128,sliding_window=None,attn_implementation='eager')
  model=MistralForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
  with torch.inference_mode():model(torch.ones((1,2),device='cuda',dtype=torch.long),use_cache=False,output_attentions=True)
  r['control']=run(model,CpuPhasePolicy);r['candidate']=run(model,BulkPayloadPolicy)
  for name in ['frames','prototype_counts','native_h2d_calls','native_d2h_calls','dispatch']:assert r['control'][name]==r['candidate'][name],name
  assert r['candidate']['bulk_steps']>0 and r['candidate']['bulk_fallback_steps']>=2 and r['candidate']['bulk_dispatches']>0
  r['partial_queue_failure']=check_partial_queue_failure(model)
  r.update(passed=True,hooks_leases_and_cache_owners_released=True)
 except BaseException as e:r['error']=f'{type(e).__name__}: {e}';raise
 finally:out.write_text(json.dumps(r,indent=2)+'\n')
 print('Bulk controller: exact7 logits/KV/slot boundaries and queued failure cleanup',flush=True)
if __name__=='__main__':main()
