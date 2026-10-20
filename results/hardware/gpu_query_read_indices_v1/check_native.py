"""Native GPU read exactness, invalid/stale guards, streams and pool switching."""
import gc,json,sys,weakref
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from direct_shared_pool import DirectSharedPool
from gpu_query_read_indices import GpuLiveReader
from provenance import atomic_json


def make(dtype,seed):
    torch.manual_seed(seed);caches={}
    cfg=SimpleNamespace(num_heads=2,head_dim=32,device='cuda',resolved_stt_kv_dtype=dtype,resolved_kv_dtype=dtype)
    for layer in range(3):
        caches[layer]=SimpleNamespace(cfg=cfg,capacity=16,base=layer*32,_logical=[1,7,3,15,0,8],
            stt_pos=[0,1,2,3,4,5],stt_shadow=torch.tensor([False,True,False,False,True,False]),
            _prefetched=None,_last_use=None,h2d_calls=0,d2h_calls=0,host_read_bytes=0,host_write_bytes=0)
    manager=DirectSharedPool(caches[0],3)
    for layer,cache in caches.items():
        manager.attach(cache);data=torch.randn(2,2,6,32,device='cuda',dtype=dtype)
        manager.enqueue(cache,data[0],data[1],cache._logical)
    manager.flush();policy=SimpleNamespace(layer_caches=caches,manager=manager)
    reader=GpuLiveReader(policy);reader.refresh();return policy,reader


def compare(policy,reader,stream):
    stream.wait_stream(torch.cuda.current_stream())  # Caller hands off queued GPU write payloads.
    with torch.cuda.stream(stream):
        for layer,cache in policy.layer_caches.items():
            indices=torch.where(~cache.stt_shadow)[0].tolist()
            reference=policy.manager.read({layer:indices})[layer]
            before=(cache.host_read_bytes,policy.manager.h2d_calls,cache.h2d_calls)
            candidate=reader.read_live(layer)
            assert all(torch.equal(a,b) for a,b in zip(reference,candidate))
            assert cache.host_read_bytes-before[0]==2*2*len(indices)*32*cache.cfg.resolved_stt_kv_dtype.itemsize
            assert policy.manager.h2d_calls-before[1]==cache.h2d_calls-before[2]==1
    stream.synchronize()


def main():
    base=Path(__file__).resolve().parent;output=base/'native.json';assert not output.exists()
    report=dict(passed=False,cases=[])
    try:
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            policies=[];readers=[];roots=[]
            try:
                for seed in (0,1):
                    policy,reader=make(dtype,seed);policies.append(policy);readers.append(reader)
                    roots.append(weakref.ref(policy.manager.pool))
                for i in (0,1,0,1):compare(policies[i],readers[i],streams[i%2])
                policy=policies[0];reader=readers[0];cache=policy.layer_caches[1]
                count=policy.manager.h2d_calls;cache.stt_shadow[0]=True
                try:reader.read_live(1)
                except RuntimeError as e:assert 'stale' in str(e)
                else:raise AssertionError('Stale shadow mapping accepted')
                assert count==policy.manager.h2d_calls
                reader.refresh();compare(policy,reader,streams[1])
                cache._logical.reverse();reader.refresh()
                replacement=torch.randn(2,2,6,32,device='cuda',dtype=dtype)
                policy.manager.enqueue(cache,replacement[0],replacement[1],cache._logical)
                compare(policy,reader,streams[0])
                cache._prefetched=([],());count=policy.manager.h2d_calls
                try:reader.read_live(1)
                except RuntimeError as e:assert 'Unconsumed' in str(e)
                else:raise AssertionError('Unconsumed prefetch accepted')
                assert count==policy.manager.h2d_calls;cache._prefetched=None
                cache.stt_shadow[:]=True;reader.refresh()
                result=reader.read_live(1);assert all(t.shape==(2,0,32) for t in result)
                del result,replacement
                report['cases'].append(dict(dtype=str(dtype),exact_kv=True,stale_guard_before_dispatch=True,
                    unconsumed_guard_before_dispatch=True,two_serial_consumer_streams=True,
                    two_pool_registration_switches=True,empty_rows=True))
            finally:
                for item in readers:item.close()
                for item in policies:
                    item.manager.close()
                    for c in item.layer_caches.values():c.pool=None;c._last_use=None
                    item.layer_caches.clear();item.manager=None
                policy=reader=cache=item=c=None
                gc.collect();torch.cuda.synchronize()
                assert all(root() is None for root in roots),'Owned pinned pool roots retained'
                assert all(r.blocks is None and r.live is None and not r.versions for r in readers)
            atomic_json(output,report)
        report.update(passed=True,roots_and_metadata_released=True)
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:atomic_json(output,report)
    print('Three native dtypes exact; stale/prefetch/empty guards, serial streams, pool switching, roots/metadata release pass',flush=True)
if __name__=='__main__':main()
