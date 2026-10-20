"""Native channel copies: exact bits, accounted bytes, guards and root lifetime."""
import gc,hashlib,json,runpy,sys,weakref
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from query_channel_reads import ChannelReader,kernel
from provenance import atomic_json
BASE=Path(__file__).resolve().parent
make=runpy.run_path(str(ROOT/'results/hardware/gpu_query_read_indices_v1/check_native.py'))['make']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bits=lambda t:t.contiguous().view(torch.uint8)


def create(dtype,seed):
    policy,old=make(dtype,seed);old.close();old=None
    policy.manager.wait();torch.cuda.synchronize()
    data=torch.arange(policy.manager.pool.numel()*policy.manager.pool.element_size(),dtype=torch.int32).to(torch.uint8)
    policy.manager.pool.view(torch.uint8).reshape(-1).copy_(data)
    reader=ChannelReader(policy);reader.refresh()
    return policy,reader


def compare(policy,reader,stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer,cache in policy.layer_caches.items():
            logical=torch.where(~cache.stt_shadow)[0].tolist()
            ref=policy.manager.read({layer:logical})[layer]
            relative=torch.tensor([len(logical)-1,0],device='cuda',dtype=torch.int64)
            before=(cache.host_read_bytes,policy.manager.h2d_calls,cache.h2d_calls)
            key=reader.read_keys(layer);value=reader.read_values(layer,relative)
            assert torch.equal(bits(ref[0]),bits(key))
            assert torch.equal(bits(ref[1].index_select(1,relative)),bits(value))
            assert cache.host_read_bytes-before[0]==2*(len(logical)+2)*32*cache.cfg.resolved_stt_kv_dtype.itemsize
            assert policy.manager.h2d_calls-before[1]==cache.h2d_calls-before[2]==2
    stream.synchronize()


def guards(pool):
    results=[]
    for fault in ['channel','unpinned','pool_shape','index_dtype','index_cpu','index_noncontiguous',
                  'output_shape','output_dtype','output_noncontiguous','index_output_alias','alignment']:
        source=pool;blocks=torch.tensor([[0,1,2],[3,4,5]],device='cuda',dtype=torch.int32)
        output=torch.full((2,3,32),-999,device='cuda',dtype=pool.dtype);channel=0
        if fault=='channel': channel=2
        elif fault=='unpinned': source=torch.empty_like(pool,pin_memory=False).copy_(pool)
        elif fault=='pool_shape': source=pool.squeeze(2)
        elif fault=='index_dtype': blocks=blocks.long()
        elif fault=='index_cpu': blocks=blocks.cpu()
        elif fault=='index_noncontiguous': blocks=blocks.t()
        elif fault=='output_shape': output=output.unsqueeze(2)
        elif fault=='output_dtype': output=output.double()
        elif fault=='output_noncontiguous': output=output.transpose(0,1)
        elif fault=='index_output_alias': blocks=output.view(torch.int32).reshape(-1)[:6].reshape(2,3)
        else: output=torch.full((2*3*32+1,),-999,device='cuda',dtype=pool.dtype)[1:].reshape(2,3,32)
        before=output.clone()
        try:kernel().gather(source,blocks,output,channel)
        except RuntimeError:pass
        else:raise AssertionError('Native metadata fault accepted: '+fault)
        assert torch.equal(bits(output),bits(before)),fault
        results.append(fault)
    # Native dimension alignment must be checked independently of contiguity.
    pool_odd=torch.zeros((2,6,1,30),dtype=pool.dtype,pin_memory=True)
    output=torch.full((2,3,30),-999,device='cuda',dtype=pool.dtype)
    blocks=torch.arange(6,device='cuda',dtype=torch.int32).reshape(2,3)
    before=output.clone()
    try:kernel().gather(pool_odd,blocks,output,0)
    except RuntimeError:pass
    else:raise AssertionError('Non16byte channel rows accepted')
    assert torch.equal(bits(output),bits(before))
    return results+['row_alignment']


def main():
    output=BASE/'native_v2.json';assert not output.exists()
    report=dict(passed=False,cases=[])
    try:
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            policies=[];readers=[];roots=[]
            try:
                for seed in (0,1):
                    policy,reader=create(dtype,seed);policies.append(policy);readers.append(reader)
                    roots.append(weakref.ref(policy.manager.pool))
                for index in (0,1,0,1):compare(policies[index],readers[index],streams[index%2])
                policy=policies[0];reader=readers[0];cache=policy.layer_caches[1]
                guard_cases=guards(policy.manager.pool)
                count=policy.manager.h2d_calls;cache.stt_shadow[0]=True
                try:reader.read_keys(1)
                except RuntimeError as e:assert 'stale' in str(e)
                else:raise AssertionError('Stale map accepted')
                assert count==policy.manager.h2d_calls
                reader.refresh();compare(policy,reader,streams[1])
                cache._logical.reverse();reader.refresh()
                replacement=torch.randn(2,2,6,32,device='cuda',dtype=dtype)
                policy.manager.enqueue(cache,replacement[0],replacement[1],cache._logical)
                compare(policy,reader,streams[0])
                cache._prefetched=([],());count=policy.manager.h2d_calls
                try:reader.read_values(1,torch.tensor([0],device='cuda'))
                except RuntimeError as e:assert 'Unconsumed' in str(e)
                else:raise AssertionError('Unconsumed prefetch accepted')
                assert count==policy.manager.h2d_calls;cache._prefetched=None
                cache.stt_shadow[:]=True;reader.refresh();count=policy.manager.h2d_calls
                key=reader.read_keys(1);value=reader.read_values(1,torch.empty(0,device='cuda',dtype=torch.long))
                assert key.shape==value.shape==(2,0,32) and count==policy.manager.h2d_calls
                del key,value,replacement
                report['cases'].append(dict(dtype=str(dtype),exact_channel_bits=True,accounted_channel_bytes=True,
                    metadata_guards=guard_cases,stale_prefetch_guards=True,serial_stream_handoff=True,
                    alternating_pool_local_mapping=True,empty_reads_no_dispatch=True))
            finally:
                for item in readers:item.close()
                for item in policies:
                    item.manager.close()
                    for c in item.layer_caches.values():c.pool=None;c._last_use=None
                    item.layer_caches.clear();item.manager=None
                policy=reader=cache=item=c=None
                gc.collect();torch.cuda.synchronize()
                assert all(root() is None for root in roots),'Owned pinned roots retained'
                assert all(r.blocks is None and r.live is None and r.key_rows==r.value_rows==0 for r in readers)
            atomic_json(output,report)
        report.update(passed=True,roots_and_metadata_released=True,
            binary_sha256=sha(Path(kernel().__file__)),binary_path=kernel().__file__,
            native_source_sha256={name:sha(ROOT/'experiments'/name) for name in ['native_channel_gather.cpp','native_channel_gather.cu']},
            torch_version=str(torch.__version__),python_major_minor=list(sys.version_info[:2]),
            limits='Trusted GPU numerical indices from validated source_maps/topk. Kernel bounds assertion guards host memory; invalid numerical indices are not tested in this shared context. Stream cases require explicit caller handoff; no parallel serving claim.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:atomic_json(output,report)
    print('Three dtypes exact channel bits/bytes,36native guards,stale/empty/stream/pool/root checks passed',flush=True)
if __name__=='__main__':main()
