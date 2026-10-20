"""Verify and summarize the preregistered document-disjoint actual-query run."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np

import longbench_eval as bench
from provenance import atomic_json, runtime_identity
from make_document_holdout import normalized


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "results/baseline_fidelity/real_query_document_holdout_20260929"
OLD_DIR = ROOT / "results/baseline_fidelity/real_query_holdout_20260929"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Two-sided bootstrap tail probability")
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between zero and one")
    directory = BASE_DIR
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["prior_manifest_sha256"] == [
        sha(folder / "manifest.json") for folder in (OLD_DIR, OLD_DIR / "batch2")]
    base_path, cand_path = directory / "control.json", directory / "candidate.json"
    base, cand = [json.loads(p.read_text()) for p in (base_path, cand_path)]
    ac, bc = base["config"], cand["config"]
    assert ac["real_query_prefill"] is False and bc["real_query_prefill"] is True
    assert {k: v for k, v in ac.items() if k != "real_query_prefill"} == {
        k: v for k, v in bc.items() if k != "real_query_prefill"}
    assert ac["runtime"] == runtime_identity()
    assert set(base["results"]) == set(cand["results"]) == set(manifest["tasks"])

    task_rows, all_diffs, task_boot = {}, [], []
    regressions = []
    for task, metadata in manifest["tasks"].items():
        data_path = Path(metadata["slice_file"])
        source_path = Path(metadata["source_file"])
        assert sha(source_path) == metadata["source_sha256"]
        assert sha(data_path) == metadata["slice_sha256"]
        rows = bench.read_jsonl(data_path)
        source_rows = bench.read_jsonl(source_path)
        assert len(source_rows) == metadata["source_row_count"]
        assert rows == [source_rows[i] for i in metadata["selected_zero_based_indices"]]
        a = base["results"][task]["tieredkv"]
        b = cand["results"][task]["tieredkv"]
        assert a["n"] == b["n"] == len(rows) == 4
        prior = [json.loads((folder / "manifest.json").read_text())["tasks"][task]
                 for folder in (OLD_DIR, OLD_DIR / "batch2")]
        excluded = set(range(prior[0]["excluded_prior_prefix_count"]))
        for old in prior:
            excluded.update(old["selected_zero_based_indices"])
        used_contexts = {hashlib.sha256(normalized(source_rows[i]["context"]).encode()).hexdigest()
                         for i in excluded}
        new_contexts = [hashlib.sha256(normalized(row["context"]).encode()).hexdigest()
                        for row in rows]
        assert len(set(new_contexts)) == 4
        assert not set(new_contexts) & used_contexts
        assert new_contexts == metadata["selected_normalized_context_sha256"]
        assert [hashlib.sha256(row["context"].encode()).hexdigest() for row in rows] == metadata["selected_context_sha256"]
        assert [x["sample_id"] for x in a["samples"]] == [
            x["sample_id"] for x in b["samples"]]
        diffs = []
        for record_a, record_b, item, expected_id in zip(
                a["samples"], b["samples"], rows, metadata["selected_sample_ids"]):
            sample_id = hashlib.sha256(
                json.dumps(item, sort_keys=True).encode()).hexdigest()
            assert sample_id == expected_id == record_a["sample_id"]
            assert 0 < record_a["final_resident_kv_payload_bytes"] <= 512 * 1024**2
            assert 0 < record_b["final_resident_kv_payload_bytes"] <= 512 * 1024**2
            for record in (record_a, record_b):
                pred = record["prediction"]
                if task in ("trec", "triviaqa", "samsum", "lsht"):
                    pred = pred.lstrip("\n").split("\n")[0]
                score = 100 * max(bench.DATASET_TO_METRIC[task](
                    pred, answer, all_classes=item.get("all_classes", []))
                    for answer in item["answers"])
                assert abs(score - record["score"]) < 1e-8
            diff = record_b["score"] - record_a["score"]
            diffs.append(diff)
            all_diffs.append({"task": task, "sample_id": sample_id,
                              "control": record_a["score"],
                              "candidate": record_b["score"], "difference": diff})
            if diff < 0:
                regressions.append({"task": task, "sample_id": sample_id,
                                    "difference": diff})
        task_rows[task] = {
            "n": len(diffs),
            "control_mean": float(np.mean([x["score"] for x in a["samples"]])),
            "candidate_mean": float(np.mean([x["score"] for x in b["samples"]])),
            "mean_difference": float(np.mean(diffs)),
            "prefill_s_median_control": float(np.median([x["prefill_s"] for x in a["samples"]])),
            "prefill_s_median_candidate": float(np.median([x["prefill_s"] for x in b["samples"]])),
            "peak_bytes_median_control": int(np.median([x["prefill_peak_alloc_bytes"] for x in a["samples"]])),
            "peak_bytes_median_candidate": int(np.median([x["prefill_peak_alloc_bytes"] for x in b["samples"]])),
            "paid_writes_per_decode_step_control": a["cache_stats"]["paid_writes"] / max(1, a["cache_stats"]["decode_steps"]),
            "paid_writes_per_decode_step_candidate": b["cache_stats"]["paid_writes"] / max(1, b["cache_stats"]["decode_steps"]),
        }
        task_boot.append(np.asarray(diffs, dtype=np.float64))

    rng = np.random.default_rng(20260929)
    draws = np.mean(np.stack([
        rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
        for d in task_boot]), axis=0)
    report = {
        "experiment": "REAL-Q-LRU-DOCUMENT-HOLDOUT",
        "method": "paired bootstrap over four distinct document groups within each task, then equal-weight mean across six tasks; one question per document group",
        "uncertainty_scope": "24 deterministically selected LongBench rows in distinct exact and casefold/whitespace-normalized context groups, with no such groups in the earlier development prefix or two holdouts; six fixed tasks, one model and seed. Semantic near duplicates and model-training overlap remain unverified.",
        "task_balanced_mean_difference": float(np.mean([x["mean_difference"] for x in task_rows.values()])),
        "bootstrap_alpha": args.alpha,
        "paired_bootstrap_interval": np.quantile(
            draws, [args.alpha / 2, 1 - args.alpha / 2]).tolist(),
        "regressions": {"n": len(regressions), "fraction": len(regressions) / len(all_diffs),
                        "over_10_points": sum(x["difference"] < -10 for x in regressions),
                        "rows": regressions},
        "per_task": task_rows,
        "samples": all_diffs,
        "inputs": {"manifest_sha256": sha(manifest_path),
                   "control_sha256": sha(base_path), "candidate_sha256": sha(cand_path),
                   "runtime": runtime_identity()},
        "decision": "go" if np.quantile(draws, args.alpha / 2) > 0 else "no-go/inconclusive",
        "limitations": "Only four documents per task; bootstrap is conditional on the selected tasks/model and may be unstable. Latency and peak summaries are descriptive across sequential runs, not a paired systems benchmark. No physical-tier or SOTA claim.",
    }
    output = directory / "report.json"
    atomic_json(output, report)
    print(json.dumps({k: report[k] for k in (
        "task_balanced_mean_difference", "paired_bootstrap_interval",
        "regressions", "per_task", "decision")}, indent=2))


if __name__ == "__main__":
    main()
