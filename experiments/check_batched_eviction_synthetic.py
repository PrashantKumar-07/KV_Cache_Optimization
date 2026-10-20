"""Cross-layer eviction parity across boundaries and multiple appended tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from batched_score_policy import ScoreBatchPolicy
from direct_shared_pool import DirectScorePolicy
from batched_eviction_policy import BatchedEvictionPolicy
from fused_eviction_policy import FusedEvictionPolicy
from check_cpu_slots import check_state
from provenance import atomic_json


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fused',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=dict(passed=False,fused=args.fused,checks=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,('experiments/check_batched_eviction_synthetic.py','experiments/batched_eviction_policy.py','experiments/direct_shared_pool.py','src/tiered_kv_cache.py','experiments/fused_eviction_policy.py','experiments/fused_hot_compaction.py','experiments/native_hot_compact.cpp','experiments/native_hot_compact.cu'))})
    policies=[]
    try:
        torch.manual_seed(0)
        if args.fused:
            from fused_hot_compaction import kernel
            report['binary_sha256']=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest()
        for inclusive in (True,False):
            for fill in (.75,1.):
                for length in (1,31,32,33,95,96,97,256):
                    policies=[cls(sram_budget=32,stt_budget=64,start_size=2,recent_size=4,
                        stt_expose_quota=8,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=fill,
                        inclusive=inclusive) for cls in (DirectScorePolicy,FusedEvictionPolicy if args.fused else BatchedEvictionPolicy)]
                    initial=tuple(tuple(torch.randn(1,8,length,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)) for _ in range(2))
                    visible=[policy(initial) for policy in policies]
                    for layer in range(2):check_state(policies[0].layer_caches[layer],policies[1].layer_caches[layer])
                    for step in range(12):
                        n_new=3 if step%4==3 else 1
                        added=tuple(tuple(torch.randn(1,8,n_new,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)) for _ in range(2))
                        pasts=[tuple(tuple(torch.cat((v,n),dim=2) for v,n in zip(pair,new)) for pair,new in zip(past,added)) for past in visible]
                        width=pasts[0][0][0].shape[2]
                        # Existing score updater consumes a last-query summary, not chunk attention.
                        weights=tuple(torch.rand(1,32,1,width,device='cuda',dtype=torch.float32) for _ in range(2))
                        for policy,past in zip(policies,pasts):policy.update_scores(past,weights)
                        visible=[policy(past) for policy,past in zip(policies,pasts)]
                        for layer in range(2):check_state(policies[0].layer_caches[layer],policies[1].layer_caches[layer])
                        assert policies[0]._live_totals()==policies[1]._live_totals()
                    outgoing=[list(policy.layer_caches.values()) for policy in policies]
                    fused_count=sum(getattr(c,'fused_compactions',0) for c in outgoing[1])
                    eager_count=sum(getattr(c,'eager_compactions',0) for c in outgoing[1])
                    assert all(getattr(c,'pending_keep',None) is None for c in outgoing[1])
                    for policy in policies:policy.reset();policy.reset();assert policy.manager is None and not policy.layer_caches
                    assert all(c.pool is None and c.manager is None for caches in outgoing for c in caches)
                    policies=[]
                    report['checks'].append(dict(inclusive=inclusive,fill=fill,prompt_tokens=length,layers=2,decode_calls=12,appended_tokens=18,last_query_attention_summary=True,exact_state_payload_counters_and_cleanup=True,fused_compactions=fused_count,eager_compactions=eager_count))
        report['passed']=True
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        for policy in policies:policy.reset()
        atomic_json(args.output,report)
    print('passes',len(report['checks']),flush=True)


if __name__=='__main__':main()
