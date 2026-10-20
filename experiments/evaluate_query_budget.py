#!/usr/bin/env python3
"""Evaluate the frozen true-query selector at explicit KV capacities."""
import argparse
import hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import longbench_eval as bench
from evaluate_query_selection import QuerySelector
from provenance import atomic_json, resume_results, runtime_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="../models/mistral-7b-instruct")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--max-samples", type=int, required=True)
    p.add_argument("--vram", type=int, required=True)
    p.add_argument("--stt", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    config = {"model": args.model, "data_dir": args.data_dir,
        "tasks": args.tasks, "max_samples": args.max_samples,
        "strategy": "query", "seed": 0, "max_ctx": 31500,
        "tiered_vram_budget": args.vram, "tiered_stt_budget": args.stt,
        "recent": 128, "stt_expose_quota": 32,
        "sttram_bifurcation_frac": .75, "kv_dtype": "bfloat16",
        "runtime": runtime_identity(),
        "selector_sha256": hashlib.sha256(Path(__file__).with_name(
            "evaluate_query_selection.py").read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_sha256": {t: hashlib.sha256(bench.task_data_path(args.data_dir,t).read_bytes()).hexdigest()
                           for t in args.tasks}}
    results = resume_results(args.output, config)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for task in args.tasks:
        if task in results:
            continue
        policy = bench.TieredKVPolicy(sram_budget=args.vram, stt_budget=args.stt,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
            sttram_bifurcation_frac=.75)
        selector = QuerySelector(model, policy)
        try:
            score, sample_scores = bench.evaluate_task(model, tokenizer, task,
                policy, "cuda", args.max_samples, True, max_ctx=31500,
                data_dir=args.data_dir)
        finally:
            selector.close()
        results[task] = {"tieredkv": {"score": score, "n": len(sample_scores),
            "sample_scores": sample_scores, "samples": policy.sample_records,
            "cache_stats": policy.stats()}}
        atomic_json(args.output, {"config": config, "results": results})
        print(task, args.vram, args.stt, score, flush=True)


if __name__ == "__main__":
    main()
