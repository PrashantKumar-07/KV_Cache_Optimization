"""Real multi-token fallback and return to canonical CPU native state."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import torch
from transformers import MistralConfig,MistralForCausalLM
from bulk_promotion_policy import BulkPromotionPolicy
from cpu_phase_policy import CpuPhasePolicy
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
 p=cls(sram_budget=191,stt_budget=64,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
 p.prompt_query_score_mode='chunked';p.prompt_query_chunk=128
 before=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
 lease=SharedAppendLease(model,p) if cls is CpuPhasePolicy else None
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
   if cls is CpuPhasePolicy:counts.append(dict(native=p.cpu_demote_calls,fallback=p.cpu_fallback_steps,attached=p.cpu_arena is not None))
  return dict(frames=frames,dispatch=counts,prototype_counts={n:sum(getattr(c,n) for c in p.layer_caches.values()) for n in ['host_read_bytes','host_write_bytes','host_initial_write_bytes']},native_h2d_calls=p.manager.h2d_calls,native_d2h_calls=p.manager.d2h_calls,append_fast_updates=lease.fast_updates if lease else 0,append_fallback_updates=lease.fallback_updates if lease else 0)
 finally:
  if capture:capture.close()
  if lease:lease.close()
  p.reset();past=out=None
  assert before==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
  if lease:assert not lease.handles and not lease.pending

def main():
 out=BASE/'gpu.json';assert not out.exists();r=dict(passed=False,plan_sha256=sha(BASE/'plan.json'))
 try:
  torch.manual_seed(7)
  config=MistralConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,vocab_size=128,sliding_window=None,attn_implementation='eager')
  model=MistralForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
  r['control']=run(model,BulkPromotionPolicy);r['candidate']=run(model,CpuPhasePolicy)
  for n in ['frames','prototype_counts','native_h2d_calls','native_d2h_calls']:assert r['control'][n]==r['candidate'][n],n
  assert r['candidate']['dispatch']==[dict(native=1,fallback=0,attached=True),dict(native=1,fallback=1,attached=True),dict(native=2,fallback=1,attached=True),dict(native=2,fallback=2,attached=True),dict(native=3,fallback=2,attached=True),dict(native=4,fallback=2,attached=True)]
  r.update(passed=True,hooks_leases_and_cache_owners_released=True)
 except BaseException as e:r['error']=f'{type(e).__name__}: {e}';raise
 finally:out.write_text(json.dumps(r,indent=2)+'\n')
 print('Real2/32-token append fallback and native return: exact7 boundaries',flush=True)
if __name__=='__main__':main()
