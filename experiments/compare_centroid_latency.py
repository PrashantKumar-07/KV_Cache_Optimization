"""Same-process ABBA full-loop timing: full-query versus centroid pages."""
import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_query_selection import PromptQueryCapture, QuerySelector
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity
from stable_page_index import IndexedTieredKVCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='../models/mistral-7b-instruct')
    p.add_argument('--prompt-file', default='results/remediation_v3/latency_prompt.txt')
    p.add_argument('--output', default='results/page_recall/centroid_latency_abba.json')
    p.add_argument('--blocks', type=int, default=1)
    p.add_argument('--indexed', action='store_true',
                   help='Compare experimental incrementally maintained stable pages')
    args = p.parse_args()
    if args.blocks < 1:
        p.error('--blocks must be positive')
    prompt = Path(args.prompt_file).read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    candidate = 'indexed' if args.indexed else 'centroid'
    order = ('query', candidate, candidate, 'query') * args.blocks
    results = {}
    for i, name in enumerate(order):
        policy = TieredKVPolicy(sram_budget=992, stt_budget=3104,
            stt_expose_quota=32, start_size=4, recent_size=128,
            kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
        if name == 'indexed':
            policy.cache_cls = IndexedTieredKVCache
        selector = QuerySelector(model, policy,
            candidate_pages=32 if name in ('centroid', 'indexed') else 0,
            use_index=name == 'indexed')
        capture = PromptQueryCapture(model, policy)
        try:
            results[f'{i}_{name}'] = profile_one(model, tokenizer, policy, 16384,
                n_decode=50, n_warmup=1, n_repeat=2, device='cuda', prompt_text=prompt)
        finally:
            selector.close()
            capture.close()
        print(i, name, results[f'{i}_{name}']['itl_ms'], flush=True)
    atomic_json(args.output, {'config': {'model': args.model, 'order': order,
        'blocks': args.blocks, 'ctx_len': 16384, 'decode': 50,
        'candidate_pages': 32, 'page_size': 16,
        'page_mode': 'stable_incremental' if args.indexed else 'centroid_vector',
        'indexed': args.indexed,
        'prefill': 'actual_rotary_query_both_arms',
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'selector_sha256': hashlib.sha256(Path(__file__).with_name(
            'evaluate_query_selection.py').read_bytes()).hexdigest(),
        'page_sha256': hashlib.sha256(Path(__file__).with_name(
            'page_candidates.py').read_bytes()).hexdigest(),
        'page_index_sha256': (hashlib.sha256(Path(__file__).with_name(
            'stable_page_index.py').read_bytes()).hexdigest()
            if args.indexed else None),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime': runtime_identity(), 'scope': 'one 16K prompt; same-process full model+policy'
        }, 'results': results})


if __name__ == '__main__':
    main()
