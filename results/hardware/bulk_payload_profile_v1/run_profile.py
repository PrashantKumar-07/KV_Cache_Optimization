"""Warmed policy/score attribution, no resource observers in measured calls."""
import cProfile,hashlib,json,pstats,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'src'),str(R/'experiments')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from bulk_payload_policy import BulkPayloadPolicy
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from latency_profile import profile_one
from compare_cpu_batched import digest
from provenance import atomic_json,runtime_identity
B=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
profiles={k:cProfile.Profile() for k in ['policy','scores']};calls={k:0 for k in profiles}
class Diagnostic(BulkPayloadPolicy):
 def reset(self):
  self._diag_trial=getattr(self,'_diag_trial',0)+1;super().reset()
 def __call__(self,*args,**kwargs):
  if self._diag_trial==2 and self.initialized:
   calls['policy']+=1;return profiles['policy'].runcall(super().__call__,*args,**kwargs)
  return super().__call__(*args,**kwargs)
 def update_scores(self,*args,**kwargs):
  if self._diag_trial==2:
   calls['scores']+=1;return profiles['scores'].runcall(super().update_scores,*args,**kwargs)
  return super().update_scores(*args,**kwargs)
def main():
 plan=json.loads((B/'profile_plan.json').read_text());refpath=R/plan['reference'];assert sha(refpath)==plan['reference_sha256'];ref=json.loads(refpath.read_text())
 for name in ['prompt','teachers']:assert sha(R/plan[name])==plan[name+'_sha256']
 torch.manual_seed(0);model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval();tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
 policy=Diagnostic(sram_budget=649,stt_budget=1982,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75);policy._diag_trial=0
 policy.prompt_query_score_mode='chunked';policy.prompt_query_chunk=128
 hooks=[len(l.self_attn._forward_pre_hooks) for l in model.model.layers];capture=PromptQueryCapture(model,policy);lease=SharedAppendLease(model,policy);manager=None
 report=dict(passed=False,profile_instrumented=True,valid_timing_gate=False,plan_sha256=sha(B/'profile_plan.json'),runtime=runtime_identity())
 try:
  teachers=json.loads((R/plan['teachers']).read_text())['config']['teacher_token_ids']
  result=profile_one(model,tokenizer,policy,16384,n_decode=50,n_warmup=1,n_repeat=1,device='cuda',prompt_text=(R/plan['prompt']).read_text(),forced_token_ids=teachers,prefill_last_logits=True)
  state=digest(policy.layer_caches);counters=policy._live_totals();manager=policy.manager
  native=dict(host_read_bytes=sum(c.host_read_bytes for c in policy.layer_caches.values()),host_write_bytes=sum(c.host_write_bytes for c in policy.layer_caches.values()),native_h2d_calls=manager.h2d_calls,native_d2h_calls=manager.d2h_calls)
  assert state==ref['state_sha256'] and counters==ref['counters'] and result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
  for k,v in native.items():assert v==ref['prototype_counts'][k],k
  assert calls==dict(policy=50,scores=50) and policy.bulk_payload_steps==50 and policy.bulk_payload_fallback_steps==0
  report.update(passed=True,state_sha256=state,counters=counters,native=native,calls=calls,result_instrumented=result,bulk_steps=50,bulk_fallback_steps=0)
 finally:
  lease.close();capture.close();policy.reset();torch.cuda.synchronize()
  assert hooks==[len(l.self_attn._forward_pre_hooks) for l in model.model.layers]
  assert not lease.handles and not lease.pending and policy.manager is None
  if manager:assert manager.pool is None and manager.bulk_pending is None
  report['owned_hooks_leases_pool_released']=True
  for k,p in profiles.items():
   p.dump_stats(str(B/(k+'.prof')))
   with (B/(k+'.txt')).open('w') as f:pstats.Stats(p,stream=f).strip_dirs().sort_stats('cumulative').print_stats(55)
  atomic_json(B/'profile.json',report)
if __name__=='__main__':main()
