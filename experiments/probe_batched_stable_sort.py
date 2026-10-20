"""Measure exact per-layer stable sorts versus padded batched stable sorts."""
import hashlib
import inspect
import statistics
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
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
    layer_by_id = {}
    original = torch.argsort
    samples = []
    counts = {}

    def record_argsort(tensor, *args, **kwargs):
        caller = inspect.currentframe().f_back
        fn = caller.f_code.co_name
        local = caller.f_locals
        kind = None
        take = 0
        request = 0
        if fn == 'evict_and_demote':
            if local.get('cand_soft') is tensor:
                kind = 'evict_hot'
                request = local['n_over']
                take = min(request,
                           int(torch.isfinite(tensor).sum().item()))
            elif local.get('cand_hard') is tensor:
                kind = 'evict_hard_fallback'
                request = local['remaining']
                take = min(request,
                           int(torch.isfinite(tensor).sum().item()))
            elif local.get('priority') is tensor:
                kind = 'evict_shadow'
                request = take = min(local['n_over'], len(local['shadow_idx']))
            elif local.get('la') is tensor:
                kind = 'evict_live_lru'
                cache = local['self']
                floor = cache.cfg.stt_live_floor if cache.cfg.inclusive else 0
                request = take = min(local['remaining'],
                                     max(0, len(local['real_idx']) - floor))
            cache = local.get('self')
            layer = layer_by_id.get(id(cache), -1)
            step = getattr(cache, '_t', -1)
        elif fn == '_visible_tuple':
            if local.get('la') is tensor:
                kind = 'visible_cold'
                request = take = local['n_stt_uni']
            elif local.get('cache') is not None and local.get('cache').sram_cum is tensor:
                kind = 'visible_hot_fallback'
                request = take = local['n_sram_uni']
            layer = local.get('layer_idx', -1)
            step = local.get('self').decode_step_count
        else:
            layer, step = -1, -1
        result = original(tensor, *args, **kwargs)
        if kind is not None:
            descending = bool(kwargs.get('descending',
                args[1] if len(args) > 1 else False))
            samples.append({'kind': kind, 'layer': layer, 'step': step,
                'take': int(take), 'request': int(request),
                'descending': descending,
                'values': tensor.detach().clone(),
                'expected': result[:take].detach().clone()})
            counts[kind] = counts.get(kind, 0) + 1
        return result

    try:
        layer_by_id.update({id(cache): layer
                            for layer, cache in policy.layer_caches.items()})
        torch.argsort = record_argsort
        profile = profile_one(model, tokenizer, policy, 16384,
            n_decode=10, n_warmup=0, n_repeat=1, prompt_text=prompt)
    finally:
        torch.argsort = original
        capture.close()

    # Merge actual per-layer selection arrays for each step and sort phase.
    groups = {}
    for row in samples:
        groups.setdefault((row['kind'], row['step']), []).append(row)
    eligible, exact = [], True
    for key, rows in groups.items():
        rows.sort(key=lambda row: row['layer'])
        if len(rows) == 32 and len({r['descending'] for r in rows}) == 1:
            eligible.append((key, rows))
            pad_value = -float('inf') if rows[0]['descending'] else float('inf')
            batch = pad_sequence([r['values'] for r in rows], batch_first=True,
                                 padding_value=pad_value)
            order = original(batch, dim=1, descending=rows[0]['descending'],
                             stable=True)
            for i, row in enumerate(rows):
                if not torch.equal(order[i, :row['take']], row['expected']):
                    exact = False

    def individual(row):
        order = original(row['values'], descending=row['descending'], stable=True)
        take = row['take']
        if row['kind'] in ('evict_hot', 'evict_hard_fallback'):
            take = min(row['request'], int(torch.isfinite(row['values']).sum().item()))
        order[:take].tolist()

    def run_individual():
        for row in samples:
            individual(row)

    def run_batched():
        eligible_keys = {key for key, _ in eligible}
        for key, rows in groups.items():
            rows.sort(key=lambda row: row['layer'])
            if key not in eligible_keys:
                for row in rows:
                    individual(row)
                continue
            pad_value = -float('inf') if rows[0]['descending'] else float('inf')
            batch = pad_sequence([r['values'] for r in rows], batch_first=True,
                                 padding_value=pad_value)
            order = original(batch, dim=1, descending=rows[0]['descending'],
                             stable=True)
            finite_counts = torch.isfinite(batch).sum(dim=1)
            width = max((r['request'] for r in rows), default=0)
            packed = torch.cat((finite_counts[:, None], order[:, :width]), dim=1).cpu()
            for i, row in enumerate(rows):
                take = row['take']
                if row['kind'] in ('evict_hot', 'evict_hard_fallback'):
                    take = min(row['request'], int(packed[i, 0]))
                packed[i, 1:1 + take].tolist()

    def timed(fn, repeats=5):
        values = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
        return statistics.median(values)

    individual_ms = timed(run_individual)
    batched_ms = timed(run_batched)
    summary = {}
    for kind in sorted(counts):
        summary[kind] = {'calls': counts[kind],
            'complete_32_layer_groups': sum(1 for key, rows in eligible
                                            if key[0] == kind)}
    output = {'config': {'runtime': runtime_identity(), 'model': model_path,
        'ctx_len': 16384, 'profile_decode': 10, 'seed': 0,
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()},
        'capture': {'itl_ms': profile['itl_ms'], 'forward_ms': profile['forward_ms'],
                    'sort_counts': summary, 'exact_batched_prefixes': exact,
                    'eligible_groups': len(eligible)},
        'microbenchmark': {'individual_stable_prefix_ms': individual_ms,
            'batched_stable_prefix_ms': batched_ms,
            'saved_ms_over_10_steps': individual_ms - batched_ms,
            'projected_saved_ms_per_step': (individual_ms-batched_ms)/10}}
    atomic_json('results/baseline_fidelity/batched_stable_sort_probe.json', output)
    print(output['capture'])
    print(output['microbenchmark'])


if __name__ == '__main__':
    main()
