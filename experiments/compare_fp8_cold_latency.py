#!/usr/bin/env python3
"""Same-GPU ABBA timing and resident-byte check for bf16 vs FP8 cold KV."""
import argparse
import gc
import hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity


def tensor_bytes(policy, name):
    return sum(getattr(c, name).numel() * getattr(c, name).element_size()
               for c in policy.layer_caches.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="../models/mistral-7b-instruct")
    p.add_argument("--prompt-file", default="results/remediation_v3/latency_prompt.txt")
    p.add_argument("--output", default="results/fp8_cold_pilot/latency.json")
    p.add_argument("--ctx-len", type=int, default=16384)
    p.add_argument("--decode", type=int, default=20)
    args = p.parse_args()
    prompt = Path(args.prompt_file).read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    order = ("bf16", "fp8", "fp8", "bf16")
    results = {}
    for i, name in enumerate(order):
        policy = TieredKVPolicy(sram_budget=1024, stt_budget=2048,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
            stt_kv_dtype=(torch.float8_e4m3fn if name == "fp8" else None),
            sttram_bifurcation_frac=.75)
        measured = profile_one(model, tokenizer, policy, args.ctx_len,
            n_decode=args.decode, n_warmup=0, n_repeat=1,
            device="cuda", prompt_text=prompt)
        measured["resident_sram_kv_bytes"] = tensor_bytes(policy, "sram_k") + tensor_bytes(policy, "sram_v")
        measured["resident_stt_kv_bytes"] = tensor_bytes(policy, "stt_k") + tensor_bytes(policy, "stt_v")
        measured["allocated_after_profile_bytes"] = torch.cuda.memory_allocated()
        results[f"{i}_{name}"] = measured
        del policy
        gc.collect(); torch.cuda.empty_cache()
        print(i, name, measured["itl_ms"], measured["resident_stt_kv_bytes"], flush=True)
    medians = {name: float(np.median([v["itl_ms"] for k, v in results.items() if k.endswith(name)]))
               for name in ("bf16", "fp8")}
    atomic_json(args.output, {"config": {"model": args.model, "ctx_len": args.ctx_len,
        "decode": args.decode, "order": order,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "runtime": runtime_identity(), "timing_scope": "ABBA; one fixed repeat/block"},
        "medians_ms_per_token": medians, "results": results})


if __name__ == "__main__":
    main()
