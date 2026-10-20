"""Paired same-budget TieredKV pilot for frozen answer-free passage selections."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention import snapkv_importance
from longbench_eval import TieredKVPolicy, generate_with_policy
from longbench_metrics import DATASET_TO_METRIC, DATASET_TO_MAXGEN, DATASET_TO_PROMPT
from provenance import atomic_json, read_jsonl, runtime_identity
from tiered_kv_cache import TieredKVCache


DEFAULT_BASE = ROOT / "results/baseline_fidelity/question_passage_holdout_20260930"
MODEL = ROOT.parent / "models/mistral-7b-instruct"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def policy_for(selection=None):
    policy = TieredKVPolicy(
        sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
        page_size=16, promote_top_pages=2, sttram_bifurcation_frac=.75,
        kv_dtype=torch.bfloat16)
    if selection is not None:
        lo, hi = selection

        class PassageCache(TieredKVCache):
            def initial_bifurcation(self, k_all, v_all, q_all, importance_override=None):
                if importance_override is not None:
                    raise RuntimeError("Unexpected external importance override")
                base = snapkv_importance(
                    q_all[:, -self.cfg.window_size:], k_all, self.cfg.pool_kernel)
                boosted = base.clone()
                boosted[lo:hi] = base.max() + 1
                result = super().initial_bifurcation(
                    k_all, v_all, q_all, importance_override=boosted)
                self.prefill_selected_fast = sum(p in self.sram_pos for p in range(lo, hi))
                return result

        policy.cache_cls = PassageCache
    return policy


def evaluate(model, tokenizer, row, task, selection):
    prompt = "[INST] " + DATASET_TO_PROMPT[task].format(**row) + " [/INST]"
    ids = tokenizer(prompt, truncation=False).input_ids
    token_hash = sha_bytes(json.dumps(ids, separators=(",", ":")).encode())
    policy = policy_for(selection)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = generate_with_policy(
            model, tokenizer, prompt, DATASET_TO_MAXGEN[task], policy,
            "cuda", max_ctx=31500, task_name=task)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    score = max(DATASET_TO_METRIC[task](
        prediction, answer, all_classes=row.get("all_classes", []))
        for answer in row["answers"]) * 100
    caches = list(policy.layer_caches.values())
    return {
        "prediction": prediction, "score": score,
        "prompt_tokens": len(ids), "prompt_token_sha256": token_hash,
        "generated_tokens": policy.last_generation["generated_tokens"],
        "model_loop_s_excludes_selection": elapsed,
        "fast_layer_token_slots": sum(len(c.sram_pos) for c in caches),
        "slow_layer_token_slots": sum(len(c.stt_pos) for c in caches),
        "selected_passage_fast_tokens_at_prefill_by_layer": (
            None if selection is None else [c.prefill_selected_fast for c in caches]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("multifieldqa_en", "qasper"))
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    base = args.base.resolve()
    out = base / f"model_pilot_{args.task}.json"
    if out.exists():
        raise FileExistsError(f"Pilot already exists: {out}")
    manifest_path = base / "manifest.json"
    selections_path = base / "selections.json"
    manifest = json.loads(manifest_path.read_text())
    selections = json.loads(selections_path.read_text())
    if selections["manifest_sha256"] != sha_bytes(manifest_path.read_bytes()):
        raise RuntimeError("Selection manifest differs from frozen data")
    task_meta = manifest["tasks"][args.task]
    data_path = ROOT / task_meta["slice_file"]
    if sha_bytes(data_path.read_bytes()) != task_meta["slice_sha256"]:
        raise RuntimeError("Frozen data slice changed")
    rows = read_jsonl(data_path)
    picked = selections["tasks"][args.task]
    if len(rows) != 12 or len(picked) != 12:
        raise RuntimeError("Selection count mismatch")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    report = {
        "scope": "24-row pilot split by task; selection preprocessing excluded from timing",
        "task": args.task, "manifest_sha256": sha_bytes(manifest_path.read_bytes()),
        "selections_sha256": sha_bytes(selections_path.read_bytes()),
        "data_sha256": sha_bytes(data_path.read_bytes()),
        "model_config_sha256": sha_bytes((MODEL / "config.json").read_bytes()),
        "runner_sha256": sha_bytes(Path(__file__).read_bytes()),
        "runtime": runtime_identity(), "samples": [],
    }
    for row, selected in zip(rows, picked):
        sample_id = sha_bytes(json.dumps(row, sort_keys=True).encode())
        if sample_id != selected["sample_id"]:
            raise RuntimeError("Sample order or identity changed")
        chosen = selected["passage_range"]
        control = evaluate(model, tokenizer, row, args.task, None)
        candidate = evaluate(model, tokenizer, row, args.task, chosen)
        if control["prompt_token_sha256"] != candidate["prompt_token_sha256"]:
            raise RuntimeError("Paired token IDs differ")
        if control["prompt_tokens"] != selected["prompt_tokens"]:
            raise RuntimeError("Selection used different prompt tokens")
        report["samples"].append({"sample_id": sample_id, "passage_range": chosen,
                                  "control": control, "candidate": candidate})
        atomic_json(out, report)
        print(args.task, len(report["samples"]), round(control["score"], 2),
              round(candidate["score"], 2), flush=True)


if __name__ == "__main__":
    main()
