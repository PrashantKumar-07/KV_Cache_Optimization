"""Verify the frozen six-row prompt-score mix falsification pilot."""
import hashlib
import json
from pathlib import Path
from statistics import median

import longbench_eval as bench
from provenance import atomic_json


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/baseline_fidelity/prompt_score_mix_dev_20260929"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    control = json.loads((BASE / "control.json").read_text())
    candidate = json.loads((BASE / "candidate.json").read_text())
    ac, bc = control["config"], candidate["config"]
    assert ac["prompt_score_mix"] == 0.0 and bc["prompt_score_mix"] == 0.5
    assert ac["real_query_prefill"] is bc["real_query_prefill"] is True
    assert {k: v for k, v in ac.items() if k != "prompt_score_mix"} == {
        k: v for k, v in bc.items() if k != "prompt_score_mix"}
    assert ac["runtime"]["source_sha256"] == "ce91b5973693d0dfcf34b44828fba30103492ebbf320931eb2863487e1fc2bee"
    assert ac["selector_sha256"] == sha(BASE / "source_snapshot/evaluate_query_selection.py")
    assert set(control["results"]) == set(candidate["results"]) == {"qasper", "multifieldqa_en"}

    tasks = {}
    losses = []
    for task in control["results"]:
        source = bench.task_data_path(ac["data_dir"], task)
        assert sha(source) == ac["dataset_sha256"][task]
        items = bench.read_jsonl(source)[:3]
        a = control["results"][task]["tieredkv"]
        b = candidate["results"][task]["tieredkv"]
        assert a["n"] == b["n"] == len(items) == 3
        differences = []
        for item, x, y in zip(items, a["samples"], b["samples"]):
            sid = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            assert x["sample_id"] == y["sample_id"] == sid
            for record in (x, y):
                pred = record["prediction"]
                score = 100 * max(bench.DATASET_TO_METRIC[task](
                    pred, answer, all_classes=item.get("all_classes", []))
                    for answer in item["answers"])
                assert abs(score - record["score"]) < 1e-8
            diff = y["score"] - x["score"]
            differences.append(diff)
            if diff < 0:
                losses.append({"task": task, "sample_id": sid, "difference": diff})
        t0 = median(x["prefill_s"] for x in a["samples"])
        t1 = median(y["prefill_s"] for y in b["samples"])
        tasks[task] = {"n": 3, "mean_quality_difference": sum(differences) / 3,
                       "prefill_median_s_control": t0,
                       "prefill_median_s_candidate": t1,
                       "prefill_median_change_pct": (t1 / t0 - 1) * 100,
                       "peak_alloc_median_bytes_control": median(x["prefill_peak_alloc_bytes"] for x in a["samples"]),
                       "peak_alloc_median_bytes_candidate": median(x["prefill_peak_alloc_bytes"] for x in b["samples"]),
                       "paid_writes_control": a["cache_stats"]["paid_writes"],
                       "paid_writes_candidate": b["cache_stats"]["paid_writes"],
                       "decode_steps_control": a["cache_stats"]["decode_steps"],
                       "decode_steps_candidate": b["cache_stats"]["decode_steps"]}
    quality = sum(t["mean_quality_difference"] for t in tasks.values()) / len(tasks)
    decision = ("advance-to-development" if quality >= 2 and
                all(t["prefill_median_change_pct"] < 5 for t in tasks.values()) and
                all(x["difference"] > -10 for x in losses) else "reject-at-pilot")
    report = {"experiment": "PROMPT-SCORE-MIX-DEV", "task_balanced_quality_difference": quality,
              "decision": decision, "tasks": tasks, "losses": losses,
              "inputs": {"control_sha256": sha(BASE / "control.json"),
                         "candidate_sha256": sha(BASE / "candidate.json"),
                         "source_sha256": ac["runtime"]["source_sha256"],
                         "selector_sha256": ac["selector_sha256"],
                         "longbench_source_snapshot_sha256": sha(BASE / "source_snapshot/longbench_eval.py")},
              "limits": "Only six previously used development rows. Sequential single runs give descriptive prefill cost; differing generated-token counts confound total write comparison. This mix is not a novelty claim."}
    atomic_json(BASE / "report.json", report)
    print(json.dumps({k: report[k] for k in ("task_balanced_quality_difference", "decision", "tasks", "losses")}, indent=2))


if __name__ == "__main__":
    main()
