#!/usr/bin/env python3
"""Controlled same-GPU, same-prompt latency comparison of shadow policies."""
import argparse
import hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="../models/mistral-7b-instruct")
    p.add_argument("--prompt-file", default="results/remediation_v3/latency_prompt.txt")
    p.add_argument("--output", default="results/shadow_ablation_full/latency.json")
    p.add_argument("--ctx-len", type=int, default=16384)
    p.add_argument("--decode", type=int, default=50)
    args = p.parse_args()
    prompt = Path(args.prompt_file).read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = {}
    for index, name in enumerate(("inclusive", "no_shadow", "no_shadow", "inclusive")):
        policy = TieredKVPolicy(sram_budget=1024, stt_budget=2048,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
            sttram_bifurcation_frac=.75, inclusive=name == "inclusive")
        stats = profile_one(model, tokenizer, policy, args.ctx_len,
            n_decode=args.decode, n_warmup=1, n_repeat=2,
            device="cuda", prompt_text=prompt)
        results[f"{index}_{name}"] = stats
        print(index, name, stats["itl_ms"], "ms/token", flush=True)
    atomic_json(args.output, {
        "config": {"model": args.model, "ctx_len": args.ctx_len,
                   "decode": args.decode, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "order": ["inclusive", "no_shadow", "no_shadow", "inclusive"],
                   "runtime": runtime_identity(),
                   "timing_scope": "end_to_end_model_policy_token_fixed_length"},
        "results": results,
    })


if __name__ == "__main__":
    main()
