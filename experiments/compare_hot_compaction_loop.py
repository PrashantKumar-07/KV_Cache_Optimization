"""Same-process ABBA timing of boolean-mask hot K/V compaction."""
import hashlib
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
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    order = ('control', 'bool_mask', 'bool_mask', 'control') * 3
    results = {}
    for i, arm in enumerate(order):
        policy = TieredKVPolicy(sram_budget=992, stt_budget=3104,
            stt_expose_quota=32, start_size=4, recent_size=128,
            kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75,
            bool_live_gather=True, bool_hot_compaction=arm == 'bool_mask')
        capture = PromptQueryCapture(model, policy)
        try:
            stats = profile_one(model, tokenizer, policy, 16384,
                n_decode=50, n_warmup=1, n_repeat=2, prompt_text=prompt)
        finally:
            capture.close()
        stats['final_kv_payload_bytes'] = sum(x.numel() * x.element_size()
            for cache in policy.layer_caches.values()
            for x in (cache.sram_k, cache.sram_v, cache.stt_k, cache.stt_v))
        results[f'{i}_{arm}'] = stats
        print(i, arm, stats['itl_ms'], 'ms/token', flush=True)
    atomic_json('results/baseline_fidelity/hot_compaction_abba.json',
        {'config': {'model': model_path, 'ctx_len': 16384, 'decode': 50,
            'blocks': 3, 'order': order,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime': runtime_identity(), 'control': 'bool_hot_compaction=False',
            'candidate': 'bool_hot_compaction=True'}, 'results': results})


if __name__ == '__main__':
    main()
