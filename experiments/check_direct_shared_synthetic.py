"""Direct initialization boundary, policy parity and cleanup integrity."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from batched_score_policy import ScoreBatchPolicy
from direct_shared_pool import DirectScorePolicy
from check_cpu_slots import check_state
from provenance import atomic_json


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=dict(passed=False,checks=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,('experiments/check_direct_shared_synthetic.py','experiments/direct_shared_pool.py'))})
    policies=[]
    try:
        torch.manual_seed(0)
        for inclusive in (True,False):
            for fill in (.75,1.):
                for length in (1,31,32,33,95,96,97,256):
                    policies=[cls(sram_budget=32,stt_budget=64,start_size=2,recent_size=4,
                        stt_expose_quota=8,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=fill,
                        inclusive=inclusive) for cls in (ScoreBatchPolicy,DirectScorePolicy)]
                    initial=tuple(tuple(torch.randn(1,8,length,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)) for _ in range(2))
                    visible=[policy(initial) for policy in policies]
                    for layer in range(2):check_state(policies[0].layer_caches[layer],policies[1].layer_caches[layer])
                    for step in range(12):
                        added=tuple(tuple(torch.randn(1,8,1,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)) for _ in range(2))
                        pasts=[tuple(tuple(torch.cat((v,n),dim=2) for v,n in zip(pair,new)) for pair,new in zip(past,added)) for past in visible]
                        width=pasts[0][0][0].shape[2]
                        weights=tuple(torch.rand(1,32,1,width,device='cuda',dtype=torch.float32) for _ in range(2))
                        for policy,past in zip(policies,pasts):policy.update_scores(past,weights)
                        visible=[policy(past) for policy,past in zip(policies,pasts)]
                        for layer in range(2):check_state(policies[0].layer_caches[layer],policies[1].layer_caches[layer])
                        assert policies[0]._live_totals()==policies[1]._live_totals()
                    outgoing=[list(policy.layer_caches.values()) for policy in policies]
                    for policy in policies:policy.reset();policy.reset();assert policy.manager is None and not policy.layer_caches
                    assert all(c.pool is None and c.manager is None for caches in outgoing for c in caches)
                    policies=[]
                    report['checks'].append(dict(inclusive=inclusive,fill=fill,prompt_tokens=length,layers=2,decode_steps=12,exact_state_payload_counters_and_cleanup=True))
        report['passed']=True
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        for policy in policies:policy.reset()
        atomic_json(args.output,report)
    print('passes',len(report['checks']),flush=True)


if __name__=='__main__':main()
