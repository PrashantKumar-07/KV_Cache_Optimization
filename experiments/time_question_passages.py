"""Time the frozen answer-free passage selector without changing its choices."""
import hashlib
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

from evaluate_question_passages import sentence_window
from longbench_metrics import DATASET_TO_PROMPT
from provenance import atomic_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/baseline_fidelity/question_passage_holdout_20260930"
MODEL = ROOT.parent / "models/mistral-7b-instruct"


def select(tokenizer, row, task):
    prompt = "[INST] " + DATASET_TO_PROMPT[task].format(**row) + " [/INST]"
    tokenized = tokenizer(prompt, return_offsets_mapping=True, truncation=False)
    offsets = tokenized.offset_mapping
    ctx_start = prompt.index(row["context"])
    ctx_end = ctx_start + len(row["context"])
    positions = [i for i, (a, b) in enumerate(offsets)
                 if a >= ctx_start and b <= ctx_end and b > a]
    lo, hi = positions[0], positions[-1] + 1
    selected, _ = sentence_window(prompt, row["context"], offsets, lo, hi,
                                  ctx_start, row["input"])
    return list(selected)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def main():
    manifest = json.loads((BASE / "manifest.json").read_text())
    selections = json.loads((BASE / "selections.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    task_rows = {}
    for task in ("multifieldqa_en", "qasper"):
        rows = read_jsonl(ROOT / manifest["tasks"][task]["slice_file"])
        pilot = json.loads((BASE / f"model_pilot_{task}.json").read_text())
        task_rows[task] = (rows, pilot["samples"], selections["tasks"][task])
    first = task_rows["multifieldqa_en"][0][0]
    select(tokenizer, first, "multifieldqa_en")
    report = {"scope": "single warmup; one CPU selection measurement per frozen prompt",
              "tasks": {}}
    all_times, all_ratios = [], []
    for task, (rows, pilot, picked) in task_rows.items():
        timings, ratios = [], []
        for row, pair, frozen in zip(rows, pilot, picked):
            started = time.perf_counter()
            chosen = select(tokenizer, row, task)
            elapsed = time.perf_counter() - started
            if chosen != frozen["passage_range"]:
                raise RuntimeError("Timed selector changed the frozen range")
            control_s = pair["control"]["model_loop_s_excludes_selection"]
            timings.append(elapsed)
            ratios.append(elapsed / control_s)
        all_times.extend(timings)
        all_ratios.extend(ratios)
        report["tasks"][task] = {
            "n": len(timings), "median_selection_s": statistics.median(timings),
            "p95_selection_s": percentile(timings, .95),
            "median_selection_to_control_loop_ratio": statistics.median(ratios),
            "p95_selection_to_control_loop_ratio": percentile(ratios, .95),
            "selection_s_by_row": timings,
        }
    report["combined_median_ratio"] = statistics.median(all_ratios)
    report["combined_p95_ratio"] = percentile(all_ratios, .95)
    report["feasibility_gate_pass"] = (report["combined_median_ratio"] < .05
                                       and report["combined_p95_ratio"] < .10)
    report["selections_sha256"] = hashlib.sha256((BASE / "selections.json").read_bytes()).hexdigest()
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(BASE / "selection_cost_report.json", report)
    print(json.dumps({k: report[k] for k in
                      ("combined_median_ratio", "combined_p95_ratio",
                       "feasibility_gate_pass")}, indent=2))


if __name__ == "__main__":
    main()
