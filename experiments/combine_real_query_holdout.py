"""Descriptive combination of the two frozen fresh-sample tranches."""
import hashlib
import json
from pathlib import Path

import numpy as np

from provenance import atomic_json, runtime_identity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/baseline_fidelity/real_query_holdout_20260929"


def load(batch):
    directory = BASE / batch
    manifest = json.loads((directory / "manifest.json").read_text())
    control = json.loads((directory / "control.json").read_text())
    candidate = json.loads((directory / "candidate.json").read_text())
    assert control["config"]["real_query_prefill"] is False
    assert candidate["config"]["real_query_prefill"] is True
    assert control["config"]["runtime"] == candidate["config"]["runtime"] == runtime_identity()
    return directory, manifest, control, candidate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    batches = [load(""), load("batch2")]
    per_task, diff_rows, draws_by_task = {}, [], []
    sample_pairs = []
    paid_writes, decode_steps = [0, 0], [0, 0]
    inputs = []
    for directory, manifest, control, candidate in batches:
        inputs.append({name: sha(directory / name) for name in
                       ("manifest.json", "control.json", "candidate.json")})
        for task, meta in manifest["tasks"].items():
            for arm in (control, candidate):
                stats = arm["results"][task]["tieredkv"]["cache_stats"]
                idx = 0 if arm is control else 1
                paid_writes[idx] += stats["paid_writes"]
                decode_steps[idx] += stats["decode_steps"]

    tasks = list(batches[0][1]["tasks"])
    for task in tasks:
        rows_a, rows_b = [], []
        seen = set()
        for _, manifest, control, candidate in batches:
            meta = manifest["tasks"][task]
            a = control["results"][task]["tieredkv"]["samples"]
            b = candidate["results"][task]["tieredkv"]["samples"]
            expected = meta["selected_sample_ids"]
            assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b] == expected
            assert not seen.intersection(expected)
            seen.update(expected)
            rows_a.extend(a)
            rows_b.extend(b)
        diffs = np.asarray([b["score"] - a["score"]
                            for a, b in zip(rows_a, rows_b)])
        sample_pairs.extend({"task": task, "control": a, "candidate": b,
                             "difference": b["score"] - a["score"],
                             "prompt_tokens": a["prompt_tokens"],
                             "control_generated_tokens": a["generated_tokens"]}
                            for a, b in zip(rows_a, rows_b))
        per_task[task] = {"n": len(diffs), "control_mean": float(np.mean([r["score"] for r in rows_a])),
                          "candidate_mean": float(np.mean([r["score"] for r in rows_b])),
                          "mean_difference": float(diffs.mean()),
                          "regressions": int((diffs < 0).sum()),
                          "over_10_point_losses": int((diffs < -10).sum()),
                          "prefill_s_median_control": float(np.median([r["prefill_s"] for r in rows_a])),
                          "prefill_s_median_candidate": float(np.median([r["prefill_s"] for r in rows_b])),
                          "prefill_peak_bytes_median_control": int(np.median([r["prefill_peak_alloc_bytes"] for r in rows_a])),
                          "prefill_peak_bytes_median_candidate": int(np.median([r["prefill_peak_alloc_bytes"] for r in rows_b]))}
        draws_by_task.append(diffs)
        diff_rows.extend({"task": task, "sample_id": a["sample_id"],
                          "difference": float(b["score"] - a["score"])}
                         for a, b in zip(rows_a, rows_b))

    rng = np.random.default_rng(20260930)
    draws = np.mean(np.stack([
        rng.choice(d, size=(20000, len(d)), replace=True).mean(axis=1)
        for d in draws_by_task]), axis=0)

    def slices(key, ranges):
        result = {}
        for name, low, high in ranges:
            rows = [r for r in sample_pairs if low <= r[key] <= high]
            diffs = [r["difference"] for r in rows]
            task_means = [float(np.mean([r["difference"] for r in rows
                                         if r["task"] == task]))
                          for task in tasks
                          if any(r["task"] == task for r in rows)]
            result[name] = {"n": len(rows),
                "mean_sample_difference": float(np.mean(diffs)) if diffs else None,
                "task_balanced_difference": float(np.mean(task_means)) if task_means else None,
                "regressions": sum(d < 0 for d in diffs),
                "over_10_point_losses": sum(d < -10 for d in diffs)}
        return result

    prompt_bins = [("<=4K", 0, 4096), ("4K-16K", 4097, 16384), (">16K", 16385, 10**9)]
    output_bins = [("<=32", 0, 32), ("33-128", 33, 128), (">128", 129, 10**9)]
    stress = {
        "prompt_length_tokens": slices("prompt_tokens", prompt_bins),
        "control_generated_length_tokens": slices("control_generated_tokens", output_bins),
    }
    report = {
        "method": "Descriptive pooled bootstrap over two sequentially collected, non-overlapping tranches; batch 2 is the preregistered distinct-row follow-up, not an independent-document replication",
        "n_per_task": 10,
        "task_balanced_mean_difference": float(np.mean([v["mean_difference"] for v in per_task.values()])),
        "paired_bootstrap_95pct_interval_descriptive": np.quantile(draws, [0.025, 0.975]).tolist(),
        "per_task": per_task,
        "regressions": {"n": sum(r["difference"] < 0 for r in diff_rows),
                        "over_10_points": sum(r["difference"] < -10 for r in diff_rows)},
        "stress_slices": stress,
        "paid_writes_per_decode_step": {"control": paid_writes[0] / max(1, decode_steps[0]),
                                        "candidate": paid_writes[1] / max(1, decode_steps[1])},
        "inputs": inputs,
        "runtime": runtime_identity(),
        "limitations": "Pooled interval is descriptive because tranche 2 was collected after tranche 1 failed to resolve the quality question. Batch 2 crossed its numerical row-bootstrap threshold, but context_overlap_audit.json finds shared documents; the interval does not account for document dependence and cannot establish calibrated independent replication. Same model/task family only; quality result does not establish an all-metric Pareto gain.",
    }
    out = BASE / "combined_descriptive.json"
    atomic_json(out, report)
    print(json.dumps({k: report[k] for k in (
        "task_balanced_mean_difference", "paired_bootstrap_95pct_interval_descriptive",
        "per_task", "regressions", "paid_writes_per_decode_step", "stress_slices")}, indent=2))


if __name__ == "__main__":
    main()
