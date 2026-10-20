"""Same-process ABBA timing of upstream-selector SnapKV and actual-query TieredKV."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_query_selection import PromptQueryCapture
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy, _attn_impl
from provenance import atomic_json, runtime_identity
from snapkv_adapter import SnapKVAdapter


@torch.inference_mode()
def profile_snap(model, ids, *, ctx_len=16384, n_decode=50, n_warmup=1, n_repeat=2):
    trials = []
    for trial in range(n_warmup + n_repeat):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cache = SnapKVAdapter(model, capacity=1024-n_decode, window=32)
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            positions = torch.arange(ctx_len, device=ids.device).unsqueeze(0)
            with _attn_impl(model, 'sdpa'):
                out = model(ids, past_key_values=cache, use_cache=True,
                            position_ids=positions)
            token = out.logits[:, -1:].argmax(-1)
            token.item()
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - start) * 1000
            prefill_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            steps = []
            for step in range(n_decode):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(token, past_key_values=cache, use_cache=True,
                    position_ids=torch.tensor([[ctx_len+step]], device=ids.device))
                token = out.logits[:, -1:].argmax(-1)
                token.item()
                torch.cuda.synchronize()
                steps.append((time.perf_counter() - start) * 1000)
            final_bytes = sum(x.numel() * x.element_size() for layer in cache.layers
                              for x in (layer.keys, layer.values))
            assert final_bytes == 512 * 2**20
            if trial >= n_warmup:
                trials.append({'ttft_ms': ttft_ms, 'step_ms': steps,
                    'prefill_peak_memory_gb': prefill_peak_gb,
                    'decode_peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
                    'final_kv_payload_bytes': final_bytes})
        finally:
            cache.close()
    return {'ttft_ms': float(np.median([x['ttft_ms'] for x in trials])),
            'itl_ms': float(np.median([s for x in trials for s in x['step_ms']])),
            'prefill_peak_memory_gb': max(x['prefill_peak_memory_gb'] for x in trials),
            'peak_memory_gb': max(x['decode_peak_memory_gb'] for x in trials),
            'final_kv_payload_bytes': 512 * 2**20,
            'trials': trials}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='../models/mistral-7b-instruct')
    p.add_argument('--prompt-file', default='results/remediation_v3/latency_prompt.txt')
    p.add_argument('--output', default='results/baseline_fidelity/snap_tier_loop_abba.json')
    p.add_argument('--blocks', type=int, default=1)
    args = p.parse_args()
    if args.blocks < 1:
        p.error('--blocks must be positive')
    prompt = Path(args.prompt_file).read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids
    assert ids.size(1) >= 16384
    ids = ids[:, :16384].to('cuda')
    order = ('snap', 'tiered', 'tiered', 'snap') * args.blocks
    results = {}
    for i, name in enumerate(order):
        if name == 'snap':
            result = profile_snap(model, ids)
        else:
            policy = TieredKVPolicy(sram_budget=992, stt_budget=3104,
                stt_expose_quota=32, start_size=4, recent_size=128,
                kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
            capture = PromptQueryCapture(model, policy)
            try:
                result = profile_one(model, tokenizer, policy, 16384,
                    n_decode=50, n_warmup=1, n_repeat=2, device='cuda',
                    prompt_text=prompt)
            finally:
                capture.close()
            result['final_kv_payload_bytes'] = sum(x.numel() * x.element_size()
                for cache in policy.layer_caches.values()
                for x in (cache.sram_k, cache.sram_v, cache.stt_k, cache.stt_v))
            assert result['final_kv_payload_bytes'] <= 512 * 2**20
        results[f'{i}_{name}'] = result
        print(i, name, round(result['itl_ms'], 3), 'ms/token', flush=True)
    atomic_json(args.output, {'config': {'model': args.model, 'order': order,
        'blocks': args.blocks, 'ctx_len': 16384, 'decode': 50,
        'tiered_fast_rows': 992, 'tiered_cold_rows': 3104,
        'tiered_exposed_cold_rows': 32, 'snap_prompt_rows': 974,
        'tiered_bool_live_gather': True, 'tiered_bool_hot_compaction': True,
        'snap_window': 32, 'configured_max_final_kv_mib': 512,
        'prefill_attention': 'sdpa_both',
        'decode_attention': 'eager_both; Tiered requests weights; Snap does not',
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime': runtime_identity(),
        'scope': 'one fixed real 16K prompt, 50 fixed decode forwards; complete model+policy'},
        'results': results})


if __name__ == '__main__':
    main()
