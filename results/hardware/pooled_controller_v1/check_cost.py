"""Frozen quiet3ABBA ordinaryBulk versus shared model/policy transitions."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from bulk_promotion_policy import BulkPromotionPolicy
from pooled_state_policy import PooledStatePolicy as NativeControllerPolicy
from shared_decode_transition import SharedAppendLease
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((BASE/'cost_plan.json').read_text());refs={}
    for arm in ['control','query']:
        path=ROOT/plan[arm];assert sha(path)==plan[arm+'_sha256'];refs[arm]=json.loads(path.read_text())
    source=(ROOT/'results/hardware/pool_query_selector_v1/check_timing.py').read_text()
    old='policy=BulkPromotionPolicy(sram_budget=991,';assert source.count(old)==1
    source=source.replace(old,"policy=(BulkPromotionPolicy if arm=='control' else NativeControllerPolicy)(sram_budget=991,")
    old="selector=PoolQuerySelector(model,policy) if arm=='query' else None";assert source.count(old)==1
    source=source.replace(old,"selector=None;append_lease=SharedAppendLease(model,policy) if arm=='query' else None")
    source=source.replace('if selector:selector.close()', 'if append_lease:append_lease.close()')
    source=source.replace('selected_layers=selector.selected_layers if selector else 0)', 'selected_layers=0,append_fast_updates=append_lease.fast_updates if append_lease else 0,append_fallback_updates=append_lease.fallback_updates if append_lease else 0)')
    source=source.replace("base/'timing_plan.json'","base/'cost_plan.json'").replace("base/'timing.json'","base/'cost.json'")
    scope=dict(__name__='scoped_ordinary_CPU_LRU_cost',__file__=str(Path(__file__)))
    exec(compile(source,'<existing-profiler-CPU-LRU-state-transitions>','exec'),scope)
    scope.update(BulkPromotionPolicy=BulkPromotionPolicy,NativeControllerPolicy=NativeControllerPolicy,SharedAppendLease=SharedAppendLease)
    original=scope['profile_one']
    def profile(model,tokenizer,policy,*args,**options):
        result=original(model,tokenizer,policy,*args,**options)
        arm='query' if isinstance(policy,NativeControllerPolicy) else 'control';ref=refs[arm]
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        for f in ['host_read_bytes','host_write_bytes']:assert sum(getattr(c,f) for c in policy.layer_caches.values())==ref['prototype_counts'][f]
        assert policy.manager.h2d_calls==ref['prototype_counts']['native_h2d_calls']==76
        assert policy.manager.d2h_calls==ref['prototype_counts']['native_d2h_calls']==57
        counts={}
        if arm=='query':
            counts={f:getattr(policy,f) for f in ['reserved_buffer_allocations','shared_hot_prefixes','shared_transition_bindings','bounded_native_groups','bounded_scratch_peak_bytes','controller_plan_calls','controller_fallback_plans','controller_rank_groups','controller_rank_h2d_bytes','controller_cpu_streak_array_peak']}
            counts['shared_immediate_compactions']=sum(c.shared_immediate_compactions for c in policy.layer_caches.values())
            counts['batched_deferred_compactions']=sum(c.batched_deferred_compactions for c in policy.layer_caches.values())
            counts['controller_backup_calls']=sum(c.controller_backup_calls for c in policy.layer_caches.values())
            counts['controller_address_calls']=policy.manager.controller_address_calls
            counts['controller_address_array_peak']=policy.manager.controller_address_array_peak
            assert counts==dict(reserved_buffer_allocations=32,shared_hot_prefixes=1632,shared_transition_bindings=1600,shared_immediate_compactions=0,batched_deferred_compactions=1600,bounded_native_groups=0,bounded_scratch_peak_bytes=0,controller_plan_calls=50,controller_fallback_plans=0,controller_rank_groups=51,controller_rank_h2d_bytes=5555328,controller_cpu_streak_array_peak=114816,controller_backup_calls=1600,controller_address_calls=101,controller_address_array_peak=65536)
        if arm=='query':
            arena=policy.state_arena
            counts.update(pooled_prepare_calls=policy.pooled_prepare_calls,pooled_compact_groups=policy.pooled_compact_groups,pooled_owned_gpu_bytes=arena.owned_gpu_bytes,pooled_descriptor_h2d_bytes=arena.descriptor_h2d_bytes,pooled_scratch_peak_bytes=arena.scratch_peak_bytes)
            assert counts['pooled_prepare_calls']==50 and counts['pooled_compact_groups']==200
            assert counts['pooled_owned_gpu_bytes']==417792 and counts['pooled_descriptor_h2d_bytes']==6457600 and counts['pooled_scratch_peak_bytes']==32536512
        result.update(state_sha256=state,logical_counters=counters,frozen_arm_pred_state_counters_transfers_exact=True,shared_transition_counts=counts)
        return result
    scope['profile_one']=profile;scope['main']()
if __name__=='__main__':main()
