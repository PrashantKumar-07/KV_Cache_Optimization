#!/usr/bin/env python3
"""Same-GPU ABBA full-loop timing for access-time and true-query selection."""
import argparse
import hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_query_selection import PromptQueryCapture, QuerySelector
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity, model_artifact_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="../models/mistral-7b-instruct")
    p.add_argument("--prompt-file", default="results/remediation_v3/latency_prompt.txt")
    p.add_argument("--output", default="results/query_selection_pilot/latency.json")
    p.add_argument("--vram", type=int, default=1024)
    p.add_argument("--stt", type=int, default=2048)
    p.add_argument("--compare-prefill", action="store_true",
                   help="Compare K-proxy with opt-in actual-query prompt placement")
    p.add_argument("--prefill-strategy", choices=("query", "lru"), default="query",
                   help="Decode selector held fixed in --compare-prefill mode")
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--prefill-last-logits", action="store_true")
    args = p.parse_args()
    prompt = Path(args.prompt_file).read_text()
    model_identity = model_artifact_identity(args.model)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.blocks < 1:
        raise ValueError('--blocks must be positive')
    pair = ("proxy", "actual", "actual", "proxy") if args.compare_prefill else (
        "lru", "query", "query", "lru")
    order = pair * args.blocks
    results = {}
    for i, name in enumerate(order):
        policy = TieredKVPolicy(sram_budget=args.vram, stt_budget=args.stt,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
            sttram_bifurcation_frac=.75)
        selector = QuerySelector(model, policy) if (args.compare_prefill and
            args.prefill_strategy == "query") or name == "query" else None
        capture = PromptQueryCapture(model, policy) if name == "actual" else None
        try:
            results[f"{i}_{name}"] = profile_one(model, tokenizer, policy, 16384,
                n_decode=50, n_warmup=1, n_repeat=2, device="cuda", prompt_text=prompt,
                prefill_last_logits=args.prefill_last_logits)
        finally:
            if selector:
                selector.close()
            if capture:
                capture.close()
        print(i, name, results[f"{i}_{name}"]["itl_ms"], "ms/token", flush=True)
    atomic_json(args.output, {"config": {"model": args.model, "model_identity": model_identity,
        "prefill_last_logits": args.prefill_last_logits,
        "ctx_len": 16384, "decode": 50, "order": order,
        "compare_prefill": args.compare_prefill, "blocks": args.blocks,
        "prefill_strategy": args.prefill_strategy,
        "tiered_vram_budget": args.vram, "tiered_stt_budget": args.stt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "selector_sha256": hashlib.sha256(Path(__file__).with_name(
            "evaluate_query_selection.py").read_bytes()).hexdigest(),
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": runtime_identity(),
        "timing_scope": "end_to_end_model_policy_token_fixed_length"},
        "results": results})


if __name__ == "__main__":
    main()
