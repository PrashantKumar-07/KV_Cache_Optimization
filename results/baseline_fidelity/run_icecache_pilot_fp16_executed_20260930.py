"""Run one matched-input IceCache feasibility request in an isolated environment.

Requires the released IceCache extension, M-DCI, and its pinned dependencies.
This is a baseline smoke test, not a resource-matched comparison.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from longbench_eval import NO_CHAT_TASKS
from longbench_metrics import DATASET_TO_MAXGEN, DATASET_TO_METRIC, DATASET_TO_PROMPT
from provenance import atomic_json, read_jsonl, runtime_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--icecache-root", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data-file", type=Path, required=True)
    ap.add_argument("--task", default="multifieldqa_en")
    ap.add_argument("--index", type=int, default=9)
    ap.add_argument("--expected-token-hash", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.icecache_root.resolve()
    sys.path.insert(0, str(root / "IceCache/benchmark"))
    from longbench_pred import load_model_and_tokenizer, parse_args, seed_everything

    item = read_jsonl(args.data_file)[args.index]
    prompt = DATASET_TO_PROMPT[args.task].format(**item)
    if args.task not in NO_CHAT_TASKS:
        prompt = f"[INST] {prompt} [/INST]"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    token_hash = hashlib.sha256(json.dumps(
        ids[0].tolist(), separators=(",", ":")).encode()).hexdigest()
    if token_hash != args.expected_token_hash:
        raise RuntimeError("IceCache environment token IDs differ from the frozen control")
    if ids.size(1) > 31500:
        raise ValueError("Pilot requires an untruncated prompt")

    seed_everything(0)
    release_args = parse_args([
        "--model", "mistral-7b-inst", "--icecache", "--page-size", "16",
        "--page-budgets", "16", "--n-unlimited-layers", "2",
        "--n-sink-pages", "2", "--n-win-pages", "2",
        "--n-max-bytes", str(1 << 30),
        "--n-max-cpu-bytes", str(4 << 30),
    ])
    model, _ = load_model_and_tokenizer(
        str(args.model), "mistral-7b-inst", torch.device("cuda"), release_args)
    ids = ids.to("cuda")
    mask = torch.ones_like(ids)
    torch.cuda.synchronize()
    before_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=DATASET_TO_MAXGEN[args.task],
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    prediction = tokenizer.decode(output[0, ids.size(1):], skip_special_tokens=True)
    scored_prediction = (prediction.lstrip("\n").split("\n")[0]
                         if args.task in ("trec", "triviaqa", "samsum", "lsht")
                         else prediction)
    score = max(DATASET_TO_METRIC[args.task](
        scored_prediction, answer, all_classes=item.get("all_classes", []))
        for answer in item["answers"]) * 100
    report = {
        "scope": "one matched-input, non-resource-matched IceCache feasibility request",
        "task": args.task, "index": args.index,
        "sample_id": hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest(),
        "input_tokens": ids.size(1), "input_token_sha256": token_hash,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generated_tokens": output.size(1) - ids.size(1),
        "prediction": prediction, "score": score,
        "whole_request_s": elapsed,
        "gpu_allocated_before_generation_bytes": before_bytes,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "declared_gpu_pool_capacity_bytes": release_args.n_max_bytes,
        "declared_cpu_pool_capacity_bytes": release_args.n_max_cpu_bytes,
        "model_dtype": "float16", "icecache_args": vars(release_args),
        "runtime": runtime_identity(),
        "icecache_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "icecache_adapter_sha256": sha(root / "IceCache/source/icecache/adapter/modeling.py"),
        "data_sha256": sha(args.data_file),
        "model_config_sha256": sha(args.model / "config.json"),
        "runner_sha256": sha(__file__),
    }
    atomic_json(args.output, report)
    print(json.dumps({k: report[k] for k in
                      ("task", "index", "input_tokens", "generated_tokens",
                       "score", "whole_request_s", "gpu_peak_allocated_bytes")}, indent=2))


if __name__ == "__main__":
    main()
