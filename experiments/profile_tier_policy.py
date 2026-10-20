"""Focused current-source cProfile of one actual-query TieredKV request."""
import cProfile
import pstats
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_query_selection import PromptQueryCapture
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy


def main():
    text = Path('results/remediation_v3/latency_prompt.txt').read_text()
    model_path = '../models/mistral-7b-instruct'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = TieredKVPolicy(sram_budget=992, stt_budget=3104,
        stt_expose_quota=32, start_size=4, recent_size=128,
        kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
    capture = PromptQueryCapture(model, policy)
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        result = profile_one(model, tokenizer, policy, 16384, n_decode=10,
            n_warmup=0, n_repeat=1, prompt_text=text)
        profiler.disable()
    finally:
        capture.close()
    path = Path('results/baseline_fidelity/tier_policy_16k_10.prof')
    profiler.dump_stats(path)
    with Path('results/baseline_fidelity/tier_policy_16k_10.txt').open('w') as output:
        pstats.Stats(profiler, stream=output).sort_stats('cumulative').print_stats(70)
    print('ITL', result['itl_ms'], 'forward', result['forward_ms'], 'profile', path)
    print('shadow per layer', [int(c.stt_shadow.sum()) for c in policy.layer_caches.values()])
    print('demotions per layer', [c.metrics.total_demoted for c in policy.layer_caches.values()])


if __name__ == '__main__':
    main()
