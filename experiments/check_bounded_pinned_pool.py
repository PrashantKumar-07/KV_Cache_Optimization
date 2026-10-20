"""Native bounded-pool saturation/boundary/inclusion/append integrity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from bounded_pinned_pool import BoundedPoolPolicy
from fused_bounded_pool import FusedBoundedPoolPolicy
from direct_shared_pool import DirectScorePolicy
from check_cpu_slots import check_state
from compare_cpu_batched import digest
from provenance import atomic_json


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--default-geometry',action='store_true',help='1024hot/2048cold/budget32/maxappend32 fullfill stress')
    parser.add_argument('--fused',action='store_true',help='Compose the bounded slots with fused coordinator/compaction')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=dict(passed=False,default_geometry=args.default_geometry,fused=args.fused,checks=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        map(Path,('experiments/check_bounded_pinned_pool.py','experiments/bounded_pinned_pool.py',
                  'experiments/direct_shared_pool.py','experiments/cpu_slot_cache.py','src/tiered_kv_cache.py',
                  'experiments/fused_bounded_pool.py','experiments/fused_eviction_policy.py','experiments/fused_hot_compaction.py'))})
    owned=[]
    try:
        torch.manual_seed(0)
        if args.fused:
            from fused_hot_compaction import kernel
            report['binary_sha256']=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest()
        for inclusive in (True,False):
            for fill in (.75,1.):
                for length in ((4096,) if args.default_geometry else (1,31,32,33,95,96,97,256,4096)):
                    hot,cold,budget,append_max=(1024,2048,32,32) if args.default_geometry else (991,1090,32,32) if length==4096 else (32,64,8,3)
                    slots=cold+budget+append_max
                    options=dict(sram_budget=hot,stt_budget=cold,start_size=4 if length==4096 else 2,recent_size=128 if length==4096 else 4,
                        stt_expose_quota=budget,promotion_budget=budget,kv_dtype=torch.bfloat16,
                        sttram_bifurcation_frac=fill,inclusive=inclusive)
                    owned=[DirectScorePolicy(**options),(FusedBoundedPoolPolicy if args.fused else BoundedPoolPolicy)(max_decode_append=append_max,**options)]
                    inputs=tuple(tuple(torch.randn(1,8,length,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)) for _ in range(2))
                    visible=[p(inputs) for p in owned];peak=[0]
                    for cache in owned[1].layer_caches.values():
                        original=cache.append_stt_pair
                        def traced(keys,values,cache=cache,original=original):
                            original(keys,values);peak[0]=max(peak[0],len(cache._logical))
                        cache.append_stt_pair=traced
                    for step in range(24):
                        n=append_max if step%4==0 else 1
                        extra=tuple(tuple(torch.randn(1,8,n,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)) for _ in range(2))
                        pasts=[tuple(tuple(torch.cat((v,e),2) for v,e in zip(pair,new)) for pair,new in zip(old,extra)) for old in visible]
                        width=pasts[0][0][0].shape[2]
                        attention=tuple(torch.rand(1,32,1,width,device='cuda') for _ in range(2))
                        for policy,past in zip(owned,pasts):policy.update_scores(past,attention)
                        visible=[policy(past) for policy,past in zip(owned,pasts)]
                        for i in range(2):
                            check_state(owned[0].layer_caches[i],owned[1].layer_caches[i])
                            c=owned[1].layer_caches[i]
                            assert len(set(c._logical+c._free))==len(c._logical+c._free)==slots
                        assert owned[0]._live_totals()==owned[1]._live_totals()
                    before=digest(owned[1].layer_caches);steps=owned[1].decode_step_count
                    invalid=tuple(tuple(torch.cat((v,torch.zeros(1,8,append_max+1,128,device='cuda',dtype=v.dtype)),2) for v in pair) for pair in visible[1])
                    try:owned[1](invalid)
                    except ValueError:pass
                    else:raise AssertionError('Append guard missing')
                    assert before==digest(owned[1].layer_caches) and steps==owned[1].decode_step_count
                    requests=[sum(c.reserved_host_bytes for c in policy.layer_caches.values()) for policy in owned]
                    compactions={name:sum(getattr(c,name,0) for c in owned[1].layer_caches.values())
                        for name in ('fused_compactions','eager_compactions')}
                    assert requests==[(cold+hot+1)*8192,slots*8192]
                    for policy in owned:policy.reset();policy.reset();assert policy.manager is None
                    owned=[]
                    report['checks'].append(dict(inclusive=inclusive,fill=fill,prompt_tokens=length,
                        layers=2,decode_calls=24,appended_tokens=6*append_max+18,last_query_summary=True,
                        hot=hot,cold=cold,promotion_budget=budget,max_decode_append=append_max,
                        transient_logical_peak=peak[0],slot_capacity=slots,root_request_bytes=requests,
                        exact_state_payload_counters=True,rejected_append_before_mutation=True,**compactions))
        assert max(r['transient_logical_peak'] for r in report['checks'])==(2112 if args.default_geometry else 1154)
        report.update(passed=True,full_transient_saturation_observed=True)
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        for policy in owned:policy.reset()
        atomic_json(args.output,report)
    print('passed',len(report['checks']),flush=True)


if __name__=='__main__':main()
