"""Measure whether evidence spans survive prompt KV placement in two loss cases."""
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import longbench_eval as bench
from evaluate_query_selection import PromptQueryCapture
from provenance import atomic_json, read_jsonl, runtime_identity


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/baseline_fidelity/real_query_document_holdout_20260929/data"
CASES = (("qasper", 1, "+0.58 for MRPC and +0.73 for QQP"),
         ("narrativeqa", 2, "When Alabama gets right behind Clarence"))


def main():
    model_path = ROOT / "../models/mistral-7b-instruct"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    report = {"runtime": runtime_identity(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "placement_config": {"fast_rows": 992, "cold_rows": 3104,
                                   "exposed_cold_rows": 32, "recent": 128,
                                   "prompt_max_tokens": 31500, "dtype": "bfloat16"},
              "cases": []}
    loss_report = json.loads((DATA.parent / "report.json").read_text())
    loss_ids = {row["sample_id"] for row in loss_report["regressions"]["rows"]}
    for task, row_index, phrase in CASES:
        path = DATA / f"{task}.jsonl"
        item = read_jsonl(path)[row_index]
        prompt = f"[INST] {bench.DATASET_TO_PROMPT[task].format(**item)} [/INST]"
        start = prompt.index(phrase)
        end = start + len(phrase)
        tokenized = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = tokenized.pop("offset_mapping")[0].tolist()
        full_ids = tokenized["input_ids"]
        old_indices = list(range(full_ids.size(1)))
        if len(old_indices) > 31500:
            half = 31500 // 2
            old_indices = old_indices[:half] + old_indices[-half:]
        original_evidence = [old for old, offset in enumerate(offsets)
                             if offset[0] < end and offset[1] > start]
        evidence = [new for new, old in enumerate(old_indices)
                    if offsets[old][0] < end and offsets[old][1] > start]
        case = {"task": task, "source_row_index": row_index,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_id": hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest(),
                "full_prompt_tokens": full_ids.size(1),
                "prompt_tokens": len(old_indices),
                "original_evidence_token_positions": original_evidence,
                "evidence_token_positions": evidence,
                "arms": {}}
        assert case["sample_id"] in loss_ids, "Diagnostic row is not a recorded regression"
        if not evidence:
            case["diagnosis"] = "Evidence span removed by middle truncation before cache placement"
            report["cases"].append(case)
            print(task, "evidence truncated", flush=True)
            continue
        ids = full_ids[:, old_indices].to("cuda")
        for actual in (False, True):
            policy = bench.TieredKVPolicy(sram_budget=992, stt_budget=3104,
                start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
                sttram_bifurcation_frac=.75, stt_expose_quota=32)
            capture = PromptQueryCapture(model, policy) if actual else None
            try:
                with torch.inference_mode(), bench._attn_impl(model, "sdpa"):
                    output = model(ids, use_cache=True,
                        position_ids=torch.arange(ids.size(1), device="cuda")[None])
                    policy(bench.to_tuple_kv(output.past_key_values))
                by_layer = []
                for layer, cache in policy.layer_caches.items():
                    hot, cold = set(cache.sram_pos), set(cache.stt_pos)
                    visible = set(policy._vis_stt_pos[layer])
                    by_layer.append({"layer": layer,
                        "hot": sum(i in hot for i in evidence),
                        "cold_visible": sum(i in visible for i in evidence),
                        "cold_hidden": sum(i in cold and i not in visible for i in evidence),
                        "dropped": sum(i not in hot and i not in cold for i in evidence)})
                case["arms"]["actual_query" if actual else "k_proxy"] = by_layer
            finally:
                if capture:
                    capture.close()
                del policy, output
                gc.collect()
        report["cases"].append(case)
        print(task, "done", flush=True)
    output_path = ROOT / "results/baseline_fidelity/real_query_document_holdout_20260929/placement_diagnosis.json"
    atomic_json(output_path, report)


if __name__ == "__main__":
    main()
