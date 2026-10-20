"""Existing quiet3ABBA profiler, original GPU-query concat vs suffix composition."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from gpu_query_read_indices import GpuReadQueryPolicy,GpuReadQuerySelector
from query_visible_suffix import SuffixQuerySelector
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent


def main():
    source=(ROOT/'results/hardware/pool_query_selector_v1/check_timing.py').read_text()
    old='policy=BulkPromotionPolicy(sram_budget=991,';assert source.count(old)==1
    source=source.replace(old,'policy=GpuReadQueryPolicy(sram_budget=991,')
    old="selector=PoolQuerySelector(model,policy) if arm=='query' else None";assert source.count(old)==1
    source=source.replace(old,"selector=(GpuReadQuerySelector if arm=='control' else SuffixQuerySelector)(model,policy)")
    source=source.replace("base/'timing_plan.json'","base/'cost_plan.json'").replace("base/'timing.json'","base/'cost.json'")
    old='selected_layers=selector.selected_layers if selector else 0)';assert source.count(old)==1
    source=source.replace(old,"selected_layers=selector.selected_layers if selector else 0,visible_suffix_updates=getattr(selector,'visible_suffix_updates',0))")
    scope=dict(__name__='scoped_native_query_suffix_cost',__file__=str(Path(__file__)))
    exec(compile(source,'<unchanged-profiler-visible-suffix-component>','exec'),scope)
    scope.update(GpuReadQueryPolicy=GpuReadQueryPolicy,GpuReadQuerySelector=GpuReadQuerySelector,SuffixQuerySelector=SuffixQuerySelector)
    original_profile=scope['profile_one'];plan=json.loads((BASE/'cost_plan.json').read_text())
    assert hashlib.sha256((ROOT/plan['control']).read_bytes()).hexdigest()==plan['control_sha256']
    ref=json.loads((ROOT/plan['control']).read_text())
    def profile(model,tokenizer,policy,*args,**options):
        result=original_profile(model,tokenizer,policy,*args,**options)
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        for field in ('host_read_bytes','host_write_bytes'):
            assert sum(getattr(c,field) for c in policy.layer_caches.values())==ref['prototype_counts'][field]
        assert policy.manager.h2d_calls==ref['native_h2d_calls'] and policy.manager.d2h_calls==ref['native_d2h_calls']
        reader=policy.query_reader
        assert reader.metadata_h2d_bytes==ref['index_metadata_h2d_bytes'] and reader.refreshes==51 and reader.reads==1600
        result.update(state_sha256=state,logical_counters=counters,frozen_query_state_transfers_exact=True,
            index_metadata_h2d_bytes=reader.metadata_h2d_bytes,index_refreshes=reader.refreshes,index_reads=reader.reads)
        return result
    scope['profile_one']=profile;scope['main']()
if __name__=='__main__':main()
