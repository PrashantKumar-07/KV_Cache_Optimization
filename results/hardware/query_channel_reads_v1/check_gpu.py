"""Reuse4K12 suffix ownership caller; compare exact states and changed KV payload."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from query_channel_reads import ChannelQueryPolicy,ChannelQuerySelector,CHANNEL_AST,kernel
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    native=json.loads((BASE/'native_v2.json').read_text());assert native['passed']
    assert sha(Path(kernel().__file__))==native['binary_sha256']
    source=(ROOT/'results/hardware/query_visible_suffix_v1/check_gpu.py').read_text()
    old="source=source.replace(old,'policy=GpuReadQueryPolicy(sram_budget=991,')"
    new='''source=source.replace(old,"policy=(GpuReadQueryPolicy if arm=='cpu_control' else ChannelQueryPolicy)(sram_budget=991,")'''
    assert source.count(old)==1;source=source.replace(old,new)
    old="(GpuReadQuerySelector if arm=='cpu_control' else SuffixQuerySelector)(model,policy)"
    new="(SuffixQuerySelector if arm=='cpu_control' else ChannelQuerySelector)(model,policy)"
    assert source.count(old)==1;source=source.replace(old,new)
    needle="    scope=dict(__name__='scoped_native_query_suffix_correctness'"
    insertion='''    source=source.replace("'host_read_bytes','host_write_bytes'", "'host_write_bytes'")
    source=source.replace("return dict(arm=arm,predicted_token_ids=predictions,frames=frames,selections=selections,",
        "return dict(arm=arm,predicted_token_ids=predictions,frames=frames,selections=selections, channel_key_reads=getattr(policy.query_reader,'key_reads',0),channel_value_reads=getattr(policy.query_reader,'value_reads',0),channel_key_rows=getattr(policy.query_reader,'key_rows',0),channel_value_rows=getattr(policy.query_reader,'value_rows',0),native_h2d_calls=policy.manager.h2d_calls,native_d2h_calls=policy.manager.d2h_calls,")
'''
    assert source.count(needle)==1;source=source.replace(needle,insertion+needle)
    old='scope.update(SuffixQuerySelector=SuffixQuerySelector,SUFFIX_AST=SUFFIX_AST)'
    assert source.count(old)==1
    source=source.replace(old,old[:-1]+',ChannelQueryPolicy=ChannelQueryPolicy,ChannelQuerySelector=ChannelQuerySelector)')
    source=source.replace("'selections','host_read_bytes','host_write_bytes'","'selections','host_write_bytes'")
    source=source.replace("'query_index_refreshes','query_index_reads'","'query_index_refreshes'")
    scope=dict(__name__='scoped_channel_read_native_correctness',__file__=str(Path(__file__)),
        ChannelQueryPolicy=ChannelQueryPolicy,ChannelQuerySelector=ChannelQuerySelector)
    exec(compile(source,'<unchanged-suffix-ownership-caller-channel-reads>','exec'),scope)
    scope['main']()
    raw=json.loads((BASE/'gpu.json').read_text());control,candidate=raw['arms']
    assert candidate['channel_key_reads']==candidate['channel_value_reads']==384
    assert candidate['query_index_reads']==768
    key_rows=candidate['channel_key_rows'];value_rows=candidate['channel_value_rows']
    assert key_rows==control['scanned_rows']==candidate['scanned_rows']
    assert value_rows==384*32
    row_bytes=8*128*2
    control_query=2*key_rows*row_bytes
    candidate_query=(key_rows+value_rows)*row_bytes
    common=control['host_read_bytes']-control_query
    assert candidate['host_read_bytes']==common+candidate_query
    assert candidate['native_h2d_calls']==control['native_h2d_calls']+384
    assert candidate['native_d2h_calls']==control['native_d2h_calls']
    output=BASE/'transfer_accounting.json';assert not output.exists()
    output.write_text(json.dumps(dict(verified=True,raw_sha256=sha(BASE/'gpu.json'),channel_ast=CHANNEL_AST,
        key_rows=key_rows,value_rows=value_rows,kv_row_bytes_per_layer= row_bytes,
        common_nonquery_kv_h2d_bytes=common,control_query_h2d_bytes=control_query,
        candidate_query_h2d_bytes=candidate_query,control_total_kv_h2d_bytes=control['host_read_bytes'],
        candidate_total_kv_h2d_bytes=candidate['host_read_bytes'],
        index_metadata_h2d_bytes=candidate['query_index_metadata_h2d_bytes'],
        native_binary_sha256=native['binary_sha256'],limits='Explicit copied nativeKV payload and measured indexmetadata, not physical PCIe/STT traffic or all library H2D.'),indent=2)+'\n')
    print('Exact frozen4K12 query/prefix/destination replay; channel payload accounting passes',flush=True)
if __name__=='__main__':main()
