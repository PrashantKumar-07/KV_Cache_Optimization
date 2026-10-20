"""Capture real eviction sorts and test exact top-k-prefix eligibility/cost."""
import inspect
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_query_selection import PromptQueryCapture
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity


def main():
    prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
    model_path = '../models/mistral-7b-instruct'
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = TieredKVPolicy(sram_budget=992, stt_budget=3104,
        stt_expose_quota=32, start_size=4, recent_size=128,
        kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
    capture = PromptQueryCapture(model, policy)
    original = torch.argsort
    samples = []

    def record_argsort(tensor, *args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == 'evict_and_demote' and tensor.ndim == 1:
            local = caller.f_locals
            if local.get('cand_soft') is tensor:
                take = min(local['n_over'], int(torch.isfinite(tensor).sum().item()))
                kind = 'hot'
            elif local.get('cand_hard') is tensor:
                take = min(local['remaining'],
                           int(torch.isfinite(tensor).sum().item()))
                kind = 'hot_fallback'
            elif local.get('priority') is tensor:
                take, kind = local.get('take', 0), 'shadow'
            elif local.get('la') is tensor:
                take = min(local['remaining'],
                           max(0, local['live'] - local['floor']))
                kind = 'live_lru'
            else:
                take, kind = 0, 'unknown'
            samples.append((tensor.detach().clone(), int(take), kind,
                            bool(kwargs.get('descending',
                                 args[0] if args else False))))
        return original(tensor, *args, **kwargs)

    try:
        torch.argsort = record_argsort
        profile = profile_one(model, tokenizer, policy, 16384,
            n_decode=10, n_warmup=0, n_repeat=1, prompt_text=prompt)
    finally:
        torch.argsort = original
        capture.close()

    def safe_topk(x, k, descending):
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=x.device)
        if k >= x.numel():
            return original(x, descending=descending, stable=True)[:k]
        values, indices = torch.topk(x, k, largest=not descending, sorted=True)
        tied = (values[1:] == values[:-1]).any() | ((x == values[-1]).sum() > 1)
        if bool(tied.item()):
            return original(x, descending=descending, stable=True)[:k]
        return indices

    checks = []
    for x, k, kind, descending in samples:
        if not 0 < k <= x.numel():
            continue
        expected = original(x, descending=descending, stable=True)[:k]
        vals, candidate = torch.topk(x, k, largest=not descending, sorted=True)
        exact = torch.equal(candidate, expected)
        selected_tie = bool((vals[1:] == vals[:-1]).any().item()) if k > 1 else False
        boundary_tie = int((x == vals[-1]).sum().item()) > 1
        checks.append({'kind': kind, 'size': x.numel(), 'k': k,
            'exact_topk': exact, 'selected_tie': selected_tie,
            'boundary_tie': boundary_tie, 'vector': x,
            'descending': descending})

    def timed(fn, repeats=3):
        readings = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for row in checks:
                fn(row['vector'], row['k'], row['descending'])
            torch.cuda.synchronize()
            readings.append((time.perf_counter() - start) * 1000)
        return statistics.median(readings)

    baseline_ms = timed(lambda x, k, d: original(x, descending=d, stable=True)[:k])
    candidate_ms = timed(safe_topk)
    summary = {}
    for kind in sorted({x['kind'] for x in checks}):
        group = [x for x in checks if x['kind'] == kind]
        summary[kind] = {'calls': len(group), 'exact_topk_calls': sum(x['exact_topk'] for x in group),
            'selected_tie_calls': sum(x['selected_tie'] for x in group),
            'boundary_tie_calls': sum(x['boundary_tie'] for x in group),
            'elements': sum(x['size'] for x in group)}
    atomic_json('results/baseline_fidelity/stable_topk_probe.json', {
        'config': {'runtime': runtime_identity(), 'ctx_len': 16384,
            'decode': 10, 'prompt_sha256': __import__('hashlib').sha256(prompt.encode()).hexdigest(),
            'source': 'current stable-sort control', 'warmup': 0},
        'profile': {'itl_ms': profile['itl_ms'], 'forward_ms': profile['forward_ms'],
                    'captured_sort_calls': len(samples), 'captured_nonempty_calls': len(checks)},
        'sorts': summary,
        'microbenchmark': {'stable_prefix_ms': baseline_ms,
            'guarded_topk_prefix_ms': candidate_ms,
            'candidate_over_control': candidate_ms / baseline_ms},
    })
    print('profile', profile['itl_ms'], profile['forward_ms'])
    print('sorts', summary)
    print('microbenchmark ms', baseline_ms, candidate_ms)


if __name__ == '__main__':
    main()
