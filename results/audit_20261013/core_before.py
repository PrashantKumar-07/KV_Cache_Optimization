"""CPU regression probes: exit nonzero on defects, optionally persist JSON.

Run from the repository with ../venv_kvcache/bin/python
results/audit_20261013/core_before.py --output <new-json-path>.
The same assertions are intended to pass after the audited fixes.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np
import torch
from longbench_eval import TieredKVPolicy
from native_controller import kernel
from tiered_kv_cache import TieredConfig, TieredKVCache


def rejects_invalid_initialization(prompt, fraction=1.0, window=4):
    cache = TieredKVCache(TieredConfig(num_heads=1, head_dim=2,
        sram_capacity=4, sttram_capacity=4, sink_size=2,
        window_size=window, sttram_bifurcation_frac=fraction))
    values = torch.zeros(1, prompt, 2)
    try:
        cache.initial_bifurcation(values, values, values)
    except ValueError:
        return
    raise AssertionError("Invalid configuration accepted: " + str(dict(
        prompt=prompt, fraction=fraction, window=window,
        hot_rows=len(cache.sram_pos), cold_rows=len(cache.stt_pos))))


def policy_fixture():
    policy = TieredKVPolicy(sram_budget=4, stt_budget=4, start_size=1,
        recent_size=1, promotion_budget=1, stt_expose_quota=2)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2) / 16
    visible = policy(((values, values),))
    return policy, policy.layer_caches[0], visible


def multi_query_scores(queries):
    policy, cache, visible = policy_fixture()
    length = visible[0][0].shape[2] + queries
    weights = torch.arange(1, 2 * queries * length + 1, dtype=torch.float32)
    weights = weights.reshape(1, 2, queries, length).softmax(-1)
    previous = cache.sram_cum.clone()
    policy.update_scores(visible, [weights])
    # Cache ages count forward observations, so each observation averages
    # over the query positions as well as the query heads.
    expected = weights[0].mean(0).mean(0)
    torch.testing.assert_close(cache.sram_cum,
        previous + expected[:len(cache.sram_pos)], rtol=0, atol=0)
    cold = list(policy._stt_attn[0].values())
    torch.testing.assert_close(torch.tensor(cold),
        expected[len(cache.sram_pos):visible[0][0].shape[2]], rtol=0, atol=0)


def ordinary_hysteresis():
    policy, cache, _ = policy_fixture()
    cache.stt_was_vram_resident.fill_(True)
    position = cache.stt_pos[0]
    policy._stt_attn = {0: {position: 1.0}}
    assert policy._promotion_plan(0, cache) == ([], 1)
    policy._stt_attn = {}
    assert policy._promotion_plan(0, cache) == (None, 0)
    assert torch.count_nonzero(cache.stt_peek_streak) == 0, "Empty step retained streak"
    policy._stt_attn = {0: {position: 1.0}}
    assert policy._promotion_plan(0, cache) == ([], 1), "Nonconsecutive hit promoted"


def native_hysteresis():
    frame = dict(shadow=np.zeros(2, np.bool_), resident=np.ones(2, np.bool_),
        streak=np.zeros(2, np.float32), attention=[(1.0, 0)],
        inclusive=True, floor=0)
    first = kernel().promotion([frame], 1, 2)[0]
    assert first["hits"] == [] and first["deferred"] == 1
    frame.update(streak=first["streak"], attention=[])
    empty = kernel().promotion([frame], 1, 2)[0]
    assert not np.count_nonzero(empty["streak"]), "Native empty step retained streak"
    frame.update(streak=empty["streak"], attention=[(1.0, 0)])
    last = kernel().promotion([frame], 1, 2)[0]
    assert last["hits"] == [] and last["deferred"] == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Preserve prior evidence: " + str(args.output))
    checks = {
        "short_prompt_rejects_eventual_hot_overflow": lambda: rejects_invalid_initialization(2),
        "empty_prompt_rejects_eventual_hot_overflow": lambda: rejects_invalid_initialization(0),
        "fraction_above_one_rejected": lambda: rejects_invalid_initialization(20, 1.5, 1),
        "negative_fraction_rejected": lambda: rejects_invalid_initialization(20, -0.5, 1),
        "two_query_scores": lambda: multi_query_scores(2),
        "eight_query_scores": lambda: multi_query_scores(8),
        "ordinary_consecutive_hysteresis": ordinary_hysteresis,
        "native_consecutive_hysteresis": native_hysteresis,
    }
    results = []
    for name, check in checks.items():
        try:
            check()
            results.append(dict(check=name, passed=True))
        except Exception as error:
            results.append(dict(check=name, passed=False,
                error_type=type(error).__name__, error=str(error)))
    report = dict(device="cpu", checks=results, passed=all(x["passed"] for x in results))
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
