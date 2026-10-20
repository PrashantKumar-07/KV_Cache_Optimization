"""CPU-only observations for the independent simulation/baseline review.

Run from repository root:
../venv_kvcache/bin/python results/audit_20261013/reviewer/simulation_review_probes.py

Prints current behavior. Observed pre-fix results are separately preserved in
simulation_review.json; this script does not assert that defects still exist.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / name) for name in ('experiments', 'src', 'tests')]
import torch
from longbench_eval import H2OPolicy, generate_with_policy, TieredKVPolicy, FullCachePolicy
from test_longbench_policies import _tiny_model
from test_october_audit import FixedTokenizer
from tiered_kv_cache import TieredKVCache, TieredConfig


def main():
    result = {}
    policy = H2OPolicy(budget=4, start_size=1, recent_size=1)
    keys = torch.zeros(1, 1, 4, 2)
    values = torch.arange(4).reshape(1, 1, 4, 1).float().expand(1, 1, 4, 2)
    policy.update_scores(((keys, values),), [torch.tensor([[[[0., 1., 0., 0.]]]])])
    policy(((keys, values),))
    keys = torch.zeros(1, 1, 5, 2)
    values = torch.arange(5).reshape(1, 1, 5, 1).float().expand(1, 1, 5, 2)
    policy.update_scores(((keys, values),), [torch.tensor([[[[0., 1., 0., 0., 0.]]]])])
    before = policy.cum_scores[0].tolist()
    retained = policy(((keys, values),))
    result['h2o_first_overflow'] = dict(real_scores_before=before,
        kept_token_identity=retained[0][1][0, 0, :, 0].tolist(),
        scores_after=policy.cum_scores[0].tolist())
    model = _tiny_model().to(torch.bfloat16)
    result['no_drop_identical_forward_costs'] = []
    for policy in [FullCachePolicy(), TieredKVPolicy(sram_budget=64,
        stt_budget=64, start_size=1, recent_size=2, kv_dtype=torch.bfloat16)]:
        prediction = generate_with_policy(model, FixedTokenizer(), 'prompt', 3, policy, 'cpu')
        result['no_drop_identical_forward_costs'].append(dict(
            policy=type(policy).__name__, prediction=prediction,
            modeled_energy_nj=policy.stats()['modeled_energy_nj']))
    result['standalone_empty_cold_and_footprint'] = []
    for dtype in [torch.float32, torch.bfloat16]:
        cache = TieredKVCache(TieredConfig(num_heads=1, head_dim=2,
            sram_capacity=4, sttram_capacity=0, sink_size=1, window_size=1,
            kv_dtype=dtype))
        prompt = torch.zeros(1, 2, 2)
        cache.initial_bifurcation(prompt, prompt, prompt)
        new = torch.zeros(1, 1, 2, dtype=dtype)
        cache.step(new, new, new, 2)
        step = cache.metrics.steps[-1]
        result['standalone_empty_cold_and_footprint'].append(dict(dtype=str(dtype),
            cold_rows=step.sttram_tokens, sketch_macs=step.sketch_macs,
            billed_sketch_us=step.lat_sketch_us,
            actual_hot_bytes=sum(t.numel() * t.element_size() for t in (cache.sram_k, cache.sram_v)),
            reported_hot_KiB=cache.metrics.summary()['peak_sram_KB']))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
