"""Reuse frozen same-profiler ABBA driver; verify each arm outside timing."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'experiments'))
from cpu_phase_policy import CpuPhasePolicy
from bulk_payload_policy import BulkPayloadPolicy
from shared_decode_transition import SharedAppendLease
from compare_cpu_batched import digest
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=json.loads((B/'cost_plan.json').read_text());refs={}
 for arm in ['control','query']:
  p=R/plan[arm];assert sha(p)==plan[arm+'_sha256'];refs[arm]=json.loads(p.read_text())
 helper=R/'results/hardware/pool_query_selector_v1/check_timing.py';assert sha(helper)==plan['helper_sha256']
 source=helper.read_text()
 old='policy=BulkPromotionPolicy(sram_budget=991,stt_budget=1090,';assert source.count(old)==1
 source=source.replace(old,"policy=(CpuPhasePolicy if arm=='control' else BulkPayloadPolicy)(sram_budget=649,stt_budget=1982,")
 old="selector=PoolQuerySelector(model,policy) if arm=='query' else None";assert source.count(old)==1
 source=source.replace(old,'selector=None;append_lease=SharedAppendLease(model,policy)')
 source=source.replace('if selector:selector.close()','append_lease.close()')
 source=source.replace('selected_layers=selector.selected_layers if selector else 0)','selected_layers=0,append_fast_updates=append_lease.fast_updates,append_fallback_updates=append_lease.fallback_updates)')
 source=source.replace("base/'timing_plan.json'","base/'cost_plan.json'").replace("base/'timing.json'","base/'cost.json'")
 scope=dict(__name__='scoped_bulk_payload_controller_cost',__file__=str(Path(__file__)))
 exec(compile(source,'<frozen-shared-profiler-bulk-payload>','exec'),scope)
 scope.update(CpuPhasePolicy=CpuPhasePolicy,BulkPayloadPolicy=BulkPayloadPolicy,SharedAppendLease=SharedAppendLease)
 original=scope['profile_one']
 def profile(model,tokenizer,policy,*args,**options):
  result=original(model,tokenizer,policy,*args,**options)
  arm='query' if isinstance(policy,BulkPayloadPolicy) else 'control';ref=refs[arm]
  state=digest(policy.layer_caches);counters=policy._live_totals()
  assert state==ref['state_sha256'] and counters==ref['counters']
  assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
  for f in ['host_read_bytes','host_write_bytes']:assert sum(getattr(c,f) for c in policy.layer_caches.values())==ref['prototype_counts'][f]
  assert policy.manager.h2d_calls==ref['prototype_counts']['native_h2d_calls']==76
  assert policy.manager.d2h_calls==ref['prototype_counts']['native_d2h_calls']==57
  counts=dict(cpu_append_calls=policy.cpu_append_calls,cpu_fallback_steps=policy.cpu_fallback_steps)
  assert counts==dict(cpu_append_calls=50,cpu_fallback_steps=0)
  if arm=='query':
   counts.update(bulk_payload_steps=policy.bulk_payload_steps,bulk_payload_fallback_steps=policy.bulk_payload_fallback_steps,bulk_dispatches=policy.manager.bulk_dispatches)
   assert counts['bulk_payload_steps']==50 and counts['bulk_payload_fallback_steps']==0 and counts['bulk_dispatches']==25
   assert policy.manager.bulk_pending is None
  result.update(state_sha256=state,logical_counters=counters,frozen_arm_pred_state_counters_transfers_exact=True,phase_counts=counts)
  return result
 scope['profile_one']=profile;scope['main']()
if __name__=='__main__':main()
