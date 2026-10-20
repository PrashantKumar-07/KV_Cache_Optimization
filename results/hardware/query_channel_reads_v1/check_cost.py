"""Frozen quiet3ABBA suffix paired reads versus exact channel reads."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from gpu_query_read_indices import GpuReadQueryPolicy
from query_visible_suffix import SuffixQuerySelector
from query_channel_reads import ChannelQueryPolicy,ChannelQuerySelector
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((BASE/'cost_plan.json').read_text())
    native=json.loads((BASE/'native_v2.json').read_text())
    assert sha(Path(native['binary_path']))==native['binary_sha256']
    refs={}
    for arm in ('control','query'):
        path=ROOT/plan[arm];assert sha(path)==plan[arm+'_sha256'];refs[arm]=json.loads(path.read_text())
    source=(ROOT/'results/hardware/pool_query_selector_v1/check_timing.py').read_text()
    old='policy=BulkPromotionPolicy(sram_budget=991,';assert source.count(old)==1
    source=source.replace(old,"policy=(GpuReadQueryPolicy if arm=='control' else ChannelQueryPolicy)(sram_budget=991,")
    old="selector=PoolQuerySelector(model,policy) if arm=='query' else None";assert source.count(old)==1
    source=source.replace(old,"selector=(SuffixQuerySelector if arm=='control' else ChannelQuerySelector)(model,policy)")
    source=source.replace("base/'timing_plan.json'","base/'cost_plan.json'").replace("base/'timing.json'","base/'cost.json'")
    old='selected_layers=selector.selected_layers if selector else 0)';assert source.count(old)==1
    source=source.replace(old,"selected_layers=selector.selected_layers if selector else 0,visible_suffix_updates=selector.visible_suffix_updates)")
    scope=dict(__name__='scoped_channel_cost',__file__=str(Path(__file__)))
    exec(compile(source,'<unchanged-profiler-channel-component>','exec'),scope)
    scope.update(GpuReadQueryPolicy=GpuReadQueryPolicy,ChannelQueryPolicy=ChannelQueryPolicy,
        SuffixQuerySelector=SuffixQuerySelector,ChannelQuerySelector=ChannelQuerySelector)
    original=scope['profile_one']
    def profile(model,tokenizer,policy,*args,**options):
        result=original(model,tokenizer,policy,*args,**options)
        arm='query' if isinstance(policy,ChannelQueryPolicy) else 'control';ref=refs[arm]
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        for field in ('host_read_bytes','host_write_bytes'):
            assert sum(getattr(c,field) for c in policy.layer_caches.values())==ref['prototype_counts'][field]
        assert policy.manager.h2d_calls==ref['native_h2d_calls'] and policy.manager.d2h_calls==ref['native_d2h_calls']
        reader=policy.query_reader
        assert reader.metadata_h2d_bytes==ref['index_metadata_h2d_bytes'] and reader.refreshes==51
        assert reader.reads==(3200 if arm=='query' else 1600)
        channel={f:getattr(reader,f) for f in ('key_reads','value_reads','key_rows','value_rows')} if arm=='query' else {}
        for field,value in channel.items():assert value==ref['channel_'+field]
        assert sha(Path(native['binary_path']))==native['binary_sha256']
        result.update(state_sha256=state,logical_counters=counters,frozen_query_state_transfers_exact=True,
            index_metadata_h2d_bytes=reader.metadata_h2d_bytes,index_refreshes=reader.refreshes,index_reads=reader.reads,
            channel_counts=channel,native_channel_binary_sha256=native['binary_sha256'])
        return result
    scope['profile_one']=profile;scope['main']()
if __name__=='__main__':main()
