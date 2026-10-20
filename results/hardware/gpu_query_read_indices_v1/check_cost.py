"""Scoped existing profiler3ABBA: original query reads versus GPU source maps."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from pool_query_selector import PoolQuerySelector
from gpu_query_read_indices import GpuReadQueryPolicy,GpuReadQuerySelector
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent


def main():
    template=ROOT/'results/hardware/pool_query_selector_v1/check_timing.py';source=template.read_text()
    old='policy=BulkPromotionPolicy(sram_budget=991,';assert source.count(old)==1
    source=source.replace(old,"policy=(BulkPromotionPolicy if arm=='control' else GpuReadQueryPolicy)(sram_budget=991,")
    old="selector=PoolQuerySelector(model,policy) if arm=='query' else None";assert source.count(old)==1
    source=source.replace(old,"selector=(PoolQuerySelector if arm=='control' else GpuReadQuerySelector)(model,policy)")
    source=source.replace("base/'timing_plan.json'","base/'cost_plan.json'").replace("base/'timing.json'","base/'cost.json'")
    scope=dict(__name__='scoped_GPU_read_index_query_cost',__file__=str(Path(__file__)))
    exec(compile(source,'<unchanged-profiler-with-GPU-query-read-indices>','exec'),scope)
    scope.update(GpuReadQueryPolicy=GpuReadQueryPolicy,GpuReadQuerySelector=GpuReadQuerySelector)
    original_profile=scope['profile_one']
    plan=json.loads((BASE/'cost_plan.json').read_text());ref=json.loads((ROOT/plan['control']).read_text())
    assert hashlib.sha256((ROOT/plan['control']).read_bytes()).hexdigest()==plan['control_sha256']
    def profile(model,tokenizer,policy,*args,**options):
        result=original_profile(model,tokenizer,policy,*args,**options)
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        for field in ('host_read_bytes','host_write_bytes'):
            assert sum(getattr(c,field) for c in policy.layer_caches.values())==ref['prototype_counts'][field]
        reader=getattr(policy,'query_reader',None)
        result.update(state_sha256=state,logical_counters=counters,frozen_query_state_transfers_exact=True,
            index_metadata_h2d_bytes=reader.metadata_h2d_bytes if reader else None,
            index_refreshes=reader.refreshes if reader else None,index_reads=reader.reads if reader else None)
        return result
    scope['profile_one']=profile;scope['main']()
if __name__=='__main__':main()
