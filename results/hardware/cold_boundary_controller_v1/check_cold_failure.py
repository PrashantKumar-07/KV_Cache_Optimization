"""Cleanup after new cold read submits H2D but scatter fails."""
import gc,hashlib,importlib.util,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'src'),str(R/'experiments')]
import torch
from transformers import MistralConfig,MistralForCausalLM
from cold_boundary import kernel
from cold_boundary_policy import ColdBoundaryPolicy
from shared_decode_transition import SharedAppendLease
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import _attn_impl,apply_tuple_kv_inplace,to_tuple_kv,prefill_logit_kwargs
from measure_fresh_process_memory import PinnedRoots
B=Path(__file__).resolve().parent
@torch.inference_mode()
def main():
 tracker=PinnedRoots();empty=torch.empty;torch.empty=tracker.empty;module=kernel();scatter=module.scatter;manager=None;past=out=None
 torch.manual_seed(7);config=MistralConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,vocab_size=128,sliding_window=None,attn_implementation='eager');model=MistralForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval();model(torch.ones((1,2),device='cuda',dtype=torch.long),use_cache=False,output_attentions=True)
 before=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
 p=ColdBoundaryPolicy(sram_budget=191,stt_budget=256,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75);p.prompt_query_score_mode='chunked';p.prompt_query_chunk=128;lease=SharedAppendLease(model,p);capture=PromptQueryCapture(model,p)
 try:
  ids=(torch.arange(512,device='cuda')%127)[None]
  with _attn_impl(model,'sdpa'):out=model(ids,use_cache=True,position_ids=torch.arange(512,device='cuda')[None],**prefill_logit_kwargs(model,True))
  past=out.past_key_values;apply_tuple_kv_inplace(past,p(to_tuple_kv(past)));capture.close();capture=None;out=None
  manager=p.manager;before_calls=manager.h2d_calls
  def fail(*args):
   assert manager.h2d_calls>before_calls and manager.last_use is not None and manager.bulk_pending is None
   raise RuntimeError('deliberate coldscatter after H2D')
  module.scatter=fail
  out=model(torch.tensor([[17]],device='cuda'),past_key_values=past,use_cache=True,position_ids=torch.tensor([[512]],device='cuda'),output_attentions=True)
  tup=to_tuple_kv(out.past_key_values);p.update_scores(tup,out.attentions)
  try:p(tup)
  except RuntimeError as e:assert str(e)=='deliberate coldscatter after H2D'
  else:raise AssertionError('Expected late coldscatter failure')
  assert manager.h2d_calls>before_calls and manager.last_use is not None and manager.bulk_pending is None and p.cold_boundary_steps==0
  witness=dict(h2d_submitted_before_failure=True,bulk_queue_flushed_before_read=True,cold_steps_completed=0)
 finally:
  module.scatter=scatter
  if capture:capture.close()
  lease.close();p.reset();past=out=tup=None;torch.empty=empty
 gc.collect();torch.cuda.synchronize()
 assert tracker.current==0 and manager.pool is None and manager.bulk_pending is None and p.manager is None and not p.layer_caches
 assert not lease.handles and not lease.pending and before==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
 outpath=B/'failure.json';assert not outpath.exists();outpath.write_text(json.dumps(dict(passed=True,**witness,pinned_after_cleanup_bytes=tracker.current,pinned_peak_bytes=tracker.peak,all_owned_pool_hooks_leases_released=True,plan_sha256=hashlib.sha256((B/'failure_plan.json').read_bytes()).hexdigest()),indent=2)+'\n');print('Late cold H2D failure cleanup exact')
if __name__=='__main__':main()
