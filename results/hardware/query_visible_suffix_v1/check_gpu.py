"""Unchanged4K12 native query caller, with suffix storage/prefix ownership checks."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments'))
from gpu_query_read_indices import GpuReadQueryPolicy,GpuReadQuerySelector
from query_visible_suffix import SuffixQuerySelector,SUFFIX_AST


def main():
    base=Path(__file__).resolve().parent
    source=(ROOT/'results/hardware/gpu_query_read_indices_v1/check_gpu.py').read_text()
    old="policy=(BulkPromotionPolicy if arm=='cpu_control' else GpuReadQueryPolicy)(sram_budget=991,"
    assert source.count(old)==1
    source=source.replace(old,'policy=GpuReadQueryPolicy(sram_budget=991,')
    old="selector=(PoolQuerySelector if arm=='cpu_control' else GpuReadQuerySelector)(model,policy)"
    assert source.count(old)==1
    source=source.replace(old,"selector=(GpuReadQuerySelector if arm=='cpu_control' else SuffixQuerySelector)(model,policy)")
    old="observers=[];selections=[];active={'selector':False};canonical_reads=[0]"
    source=source.replace(old,old+';ownership_checks=[0];retained_destinations=[0]')
    old="""            def checked(module,args,options,select=select):
                active['selector']=True
                try:return select(module,args,options)
                finally:active['selector']=False"""
    new="""            def checked(module,args,options,select=select,index=i):
                active['selector']=True
                try:
                    hidden=options.get('hidden_states',args[0] if args else None)
                    if index not in policy.layer_caches or hidden.shape[1]!=1:
                        return select(module,args,options)
                    cache=policy.layer_caches[index];keep=policy._vis_sram_keep[index]
                    row=options['past_key_values'].layers[index]
                    before=[(id(t),t.data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()))
                        for t in (row.keys,row.values)]
                    canonical=[cache.sram_k.clone(),cache.sram_v.clone()]
                    for visible,hot in zip((row.keys,row.values),canonical):
                        assert torch.equal(visible[0,:,:len(keep),:].contiguous().view(torch.uint8),
                            hot[:,keep,:].to(visible.dtype).contiguous().view(torch.uint8))
                    result=select(module,args,options)
                    for visible,hot,actual in zip((row.keys,row.values),canonical,(cache.sram_k,cache.sram_v)):
                        assert torch.equal(visible[0,:,:len(keep),:].contiguous().view(torch.uint8),
                            hot[:,keep,:].to(visible.dtype).contiguous().view(torch.uint8))
                        assert torch.equal(hot.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8))
                    ownership_checks[0]+=1
                    if isinstance(selector,SuffixQuerySelector):
                        after=[(id(t),t.data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()))
                            for t in (row.keys,row.values)]
                        assert before==after
                        retained_destinations[0]+=1
                    return result
                finally:active['selector']=False"""
    assert source.count(old)==1;source=source.replace(old,new)
    old="return dict(arm=arm,predicted_token_ids=predictions,frames=frames,selections=selections,"
    new=old+"\n            prefix_canonical_ownership_checks=ownership_checks[0],retained_visible_destinations=retained_destinations[0],\n            visible_suffix_updates=getattr(selector,'visible_suffix_updates',0),"
    assert source.count(old)==1;source=source.replace(old,new)
    source=source.replace('read_ast=READ_AST,','read_ast=READ_AST,suffix_ast=SUFFIX_AST,')
    scope=dict(__name__='scoped_native_query_suffix_correctness',__file__=str(Path(__file__)))
    exec(compile(source,'<unchanged4K12-native-caller-with-visible-suffix>','exec'),scope)
    scope.update(SuffixQuerySelector=SuffixQuerySelector,SUFFIX_AST=SUFFIX_AST)
    scope['main']()
    report=json.loads((base/'gpu.json').read_text());plan=json.loads((base/'plan.json').read_text())
    assert hashlib.sha256((ROOT/plan['control']).read_bytes()).hexdigest()==plan['control_sha256']
    reference=json.loads((ROOT/plan['control']).read_text())['arms'][1]
    for arm in report['arms']:
        for field in ['predicted_token_ids','frames','selections','host_read_bytes','host_write_bytes',
                      'host_initial_write_bytes','query_index_metadata_h2d_bytes','query_index_refreshes','query_index_reads']:
            assert arm[field]==reference[field],(arm['arm'],field)
        assert arm['prefix_canonical_ownership_checks']==384
    candidate=report['arms'][1]
    assert candidate['retained_visible_destinations']==candidate['visible_suffix_updates']==384
    # Separate proof avoids modifying raw evidence written by the original caller.
    output=base/'frozen_replay.json';assert not output.exists()
    output.write_text(json.dumps(dict(verified=True,raw_sha256=hashlib.sha256((base/'gpu.json').read_bytes()).hexdigest(),
        frozen_gpu_index_arm_sha256=plan['control_sha256'],all_frozen_frames_choices_and_transfer_metadata_exact=True,
        prefix_canonical_ownership_checks_per_arm=384,retained_candidate_destinations=384),indent=2)+'\n')
    print('Frozen GPU-index replay,384 retained destinations and unchanged hot-prefix/canonical KV pass',flush=True)

if __name__=='__main__':main()
