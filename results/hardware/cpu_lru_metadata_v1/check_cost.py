"""Frozen quiet3ABBA ordinaryBulk versus CPU timestamp state transitions."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from bulk_promotion_policy import BulkPromotionPolicy
from cpu_lru_metadata import CpuLruPolicy
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((BASE/'cost_plan.json').read_text());refs={}
    for arm in ['control','query']:
        path=ROOT/plan[arm];assert sha(path)==plan[arm+'_sha256'];refs[arm]=json.loads(path.read_text())
    source=(ROOT/'results/hardware/pool_query_selector_v1/check_timing.py').read_text()
    old='policy=BulkPromotionPolicy(sram_budget=991,';assert source.count(old)==1
    source=source.replace(old,"policy=(BulkPromotionPolicy if arm=='control' else CpuLruPolicy)(sram_budget=991,")
    old="selector=PoolQuerySelector(model,policy) if arm=='query' else None";assert source.count(old)==1
    source=source.replace(old,'selector=None')
    source=source.replace("base/'timing_plan.json'","base/'cost_plan.json'").replace("base/'timing.json'","base/'cost.json'")
    scope=dict(__name__='scoped_ordinary_CPU_LRU_cost',__file__=str(Path(__file__)))
    exec(compile(source,'<existing-profiler-CPU-LRU-state-transitions>','exec'),scope)
    scope.update(BulkPromotionPolicy=BulkPromotionPolicy,CpuLruPolicy=CpuLruPolicy)
    original=scope['profile_one']
    def profile(model,tokenizer,policy,*args,**options):
        result=original(model,tokenizer,policy,*args,**options)
        arm='query' if isinstance(policy,CpuLruPolicy) else 'control';ref=refs[arm]
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        for f in ['host_read_bytes','host_write_bytes']:assert sum(getattr(c,f) for c in policy.layer_caches.values())==ref['prototype_counts'][f]
        assert policy.manager.h2d_calls==ref['prototype_counts']['native_h2d_calls']==76
        assert policy.manager.d2h_calls==ref['prototype_counts']['native_d2h_calls']==57
        metadata={}
        if arm=='query':
            metadata=dict(h2d_bytes=policy.timestamp_h2d_bytes,refreshes=policy.timestamp_refreshes,initial_d2h_bytes=sum(c.timestamp_initial_d2h_bytes for c in policy.layer_caches.values()),cpu_tensor_bytes=sum(c.stt_last_access.numel()*c.stt_last_access.element_size() for c in policy.layer_caches.values()))
            assert metadata['h2d_bytes']==ref['timestamp_h2d_bytes'] and metadata['refreshes']==51
            assert metadata['initial_d2h_bytes']==ref['timestamp_initial_d2h_bytes']
            assert all(c.stt_last_access.device.type=='cpu' and not hasattr(c,'_visible_access_gpu') for c in policy.layer_caches.values())
        result.update(state_sha256=state,logical_counters=counters,frozen_arm_pred_state_counters_transfers_exact=True,timestamp_metadata=metadata,mirrors_released=True)
        return result
    scope['profile_one']=profile;scope['main']()
if __name__=='__main__':main()
