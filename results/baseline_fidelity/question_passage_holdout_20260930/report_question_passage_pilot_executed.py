"""Verify and summarize the frozen paired passage-placement pilot."""
import hashlib
import json
import random
import statistics
from pathlib import Path

from longbench_metrics import DATASET_TO_METRIC
from provenance import atomic_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/baseline_fidelity/question_passage_holdout_20260930"
TASKS = ("multifieldqa_en", "qasper")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((BASE / "manifest.json").read_text())
    selections = json.loads((BASE / "selections.json").read_text())
    manifest_hash = sha((BASE / "manifest.json").read_bytes())
    selection_hash = sha((BASE / "selections.json").read_bytes())
    if selections["manifest_sha256"] != manifest_hash:
        raise RuntimeError("Selection/manifest mismatch")
    report = {"scope": "24 paired rows; conditional fixed-task uncertainty",
              "manifest_sha256": manifest_hash, "selections_sha256": selection_hash,
              "tasks": {}}
    deltas_by_task = []
    for task in TASKS:
        raw_path = BASE / f"model_pilot_{task}.json"
        raw = json.loads(raw_path.read_text())
        meta = manifest["tasks"][task]
        rows = read_jsonl(ROOT / meta["slice_file"])
        pairs = raw["samples"]
        picked = selections["tasks"][task]
        if len(rows) != len(pairs) or len(rows) != len(picked):
            raise RuntimeError("Paired row count mismatch")
        if len(rows) != 12 or raw["manifest_sha256"] != manifest_hash \
                or raw["selections_sha256"] != selection_hash:
            raise RuntimeError("Pilot provenance mismatch")
        deltas = []
        for row, pair, selection in zip(rows, pairs, picked):
            sample_id = sha(json.dumps(row, sort_keys=True).encode())
            if sample_id != pair["sample_id"] or sample_id != selection["sample_id"]:
                raise RuntimeError("Sample identity mismatch")
            if pair["passage_range"] != selection["passage_range"]:
                raise RuntimeError("Selection changed")
            control, candidate = pair["control"], pair["candidate"]
            if control["prompt_token_sha256"] != candidate["prompt_token_sha256"]:
                raise RuntimeError("Paired prompts differ")
            if candidate["selected_passage_fast_tokens_at_prefill_by_layer"] != [128] * 32:
                raise RuntimeError("Selected passage did not enter fast cache")
            for arm in (control, candidate):
                score = max(DATASET_TO_METRIC[task](
                    arm["prediction"], answer,
                    all_classes=row.get("all_classes", [])) for answer in row["answers"]) * 100
                if abs(arm["score"] - score) > 1e-8:
                    raise RuntimeError("Stored score does not recompute")
            deltas.append(candidate["score"] - control["score"])
        deltas_by_task.append(deltas)
        report["tasks"][task] = {
            "control_mean": statistics.mean(p["control"]["score"] for p in pairs),
            "candidate_mean": statistics.mean(p["candidate"]["score"] for p in pairs),
            "paired_delta": statistics.mean(deltas),
            "wins": sum(x > 1e-8 for x in deltas),
            "losses": sum(x < -1e-8 for x in deltas),
            "losses_over_10": sum(x < -10 for x in deltas),
            "deltas": deltas,
            "raw_sha256": sha(raw_path.read_bytes()),
        }
    observed = statistics.mean(statistics.mean(x) for x in deltas_by_task)
    rng = random.Random(20260930)
    draws = sorted(statistics.mean(statistics.mean(
        rng.choices(task_deltas, k=len(task_deltas)))
        for task_deltas in deltas_by_task) for _ in range(10000))
    report["task_balanced_delta"] = observed
    report["paired_bootstrap_95_interval"] = [draws[249], draws[9749]]
    report["preregistered_quality_gate_pass"] = (observed >= 2 and all(
        x >= -10 for task_deltas in deltas_by_task for x in task_deltas))
    report["runner_sha256"] = sha(Path(__file__).read_bytes())
    atomic_json(BASE / "model_pilot_report.json", report)
    print(json.dumps({"task_balanced_delta": observed,
                      "paired_bootstrap_95_interval": report["paired_bootstrap_95_interval"],
                      "quality_gate_pass": report["preregistered_quality_gate_pass"],
                      "tasks": {t: {k: v[k] for k in
                                     ("paired_delta", "wins", "losses", "losses_over_10")}
                                for t, v in report["tasks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
