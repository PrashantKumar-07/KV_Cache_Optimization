"""Asynchronous alternating-request/stream cold-pool isolation and reset gate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from cpu_batched_cache import CpuBatchCache, SharedPool
from provenance import atomic_json, runtime_identity
from tiered_kv_cache import TieredConfig


def make_pool(tag, cfg):
    caches = {}; expected = {}
    for layer in range(2):
        cache = CpuBatchCache(cfg); caches[layer] = cache
        keys = torch.full((8,256,128), float(tag+layer), device='cuda', dtype=torch.bfloat16)
        values = keys + 32
        cache.initial_bifurcation(keys, values, keys)
        expected[layer] = torch.stack((cache.stt_k, cache.stt_v)).clone()
    return caches, SharedPool(caches), expected


def close(pool):
    caches, manager, _ = pool
    manager.close()
    for cache in caches.values(): cache.close()
    manager.close()
    assert manager.pool is None and all(c.pool is None and c.manager is None for c in caches.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(f'Preserve prior evidence: {args.output}')
    sources = ('experiments/check_cpu_multi_pool.py','experiments/cpu_batched_cache.py','experiments/cpu_slot_cache.py')
    report = dict(passed=False,runtime=runtime_identity(),checks=[],
        source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    pools = {}
    try:
        cfg = TieredConfig(num_heads=8,head_dim=128,sink_size=2,window_size=4,
            sram_capacity=32,sttram_capacity=64,page_size=4,kv_dtype=torch.bfloat16,
            device='cuda',inclusive=True,sttram_bifurcation_frac=.75)
        for requests in (2,4):
            for stream_count in (1,2):
                pools = {r:make_pool(8*r,cfg) for r in range(requests)}
                streams = [torch.cuda.Stream() for _ in range(stream_count)]
                for stream in streams: stream.wait_stream(torch.cuda.current_stream())
                observations = []; overlaps = resets = 0
                for step in range(32):
                    for request in range(requests):
                        caches,manager,expected = pools[request]
                        with torch.cuda.stream(streams[(step+request)%stream_count]):
                            # Queue a delay so read/write event ordering is exercised asynchronously.
                            torch.cuda._sleep(500000)
                            for layer,cache in caches.items():
                                logical = (step+layer)%len(cache._logical)
                                slot = cache._logical[logical]
                                tag = (request*48 + step + layer*4)%192
                                keys = torch.full((8,1,128),float(tag),device='cuda',dtype=torch.bfloat16)
                                values = keys + 32
                                manager.enqueue(cache,keys,values,[slot])
                                expected[layer][:,:,logical] = torch.stack((
                                    torch.full((8,128),float(tag),dtype=torch.bfloat16),
                                    torch.full((8,128),float(tag+32),dtype=torch.bfloat16)))
                            selections = {layer: ([] if step%7==0 else [0,0] if step%5==0 else list(range(min(3+layer,len(cache._logical))))[::-1]) for layer,cache in caches.items()}
                            outputs = manager.read(selections)
                            for layer,idx in selections.items():
                                observations.append((outputs[layer],expected[layer][:,:,idx].clone(),request,step,layer))
                            if step%8==0:
                                cache = caches[0];slot = cache._logical[0]
                                pair = torch.full((8,1,128),float(200+request),device='cuda',dtype=torch.bfloat16)
                                calls = manager.d2h_calls
                                manager.enqueue(cache,pair,pair,[slot]);manager.enqueue(cache,pair+1,pair+1,[slot])
                                assert manager.d2h_calls == calls+1
                                expected[0][:,:,0].fill_(201+request);overlaps+=1
                    if step==15:
                        caches,manager,_ = pools[0]
                        with torch.cuda.stream(streams[step%stream_count]):
                            data = torch.full((8,1,128),240.,device='cuda',dtype=torch.bfloat16)
                            manager.enqueue(caches[0],data,data,[caches[0]._logical[0]])
                            assert manager.pending
                            close(pools[0]);pools[0]=make_pool(224,cfg);resets+=1
                for stream in streams: stream.synchronize()
                for (keys,values),expected,request,step,layer in observations:
                    assert torch.equal(keys.cpu(),expected[0]),('keys',request,step,layer)
                    assert torch.equal(values.cpu(),expected[1]),('values',request,step,layer)
                for pool in pools.values(): close(pool)
                pools={}
                report['checks'].append(dict(requests=requests,layers=2,streams=stream_count,rounds=32,
                    exact_read_observations=len(observations),overlap_flushes=overlaps,queued_write_resets=resets))
                atomic_json(args.output,report)
        report.update(passed=True,limitations='Synthetic cold storage with serialized Python host dispatch and overlapping CUDA streams. No model-serving/thread safety/throughput/saturation or allocator peak claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        for pool in pools.values(): close(pool)
        atomic_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
