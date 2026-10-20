"""Verify paired teacher-forced shadow traffic and summarize ABBA costs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from provenance import atomic_json, runtime_identity


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/shadow_ablation_full/fixed_stream_20260929.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(SOURCE.read_text())
    cfg = data["config"]
    assert cfg["runtime"] == runtime_identity()
    assert cfg["profile_script_sha256"] == sha(ROOT / "experiments/latency_profile.py")
    assert cfg["driver_sha256"] == sha(ROOT / "experiments/compare_shadow_fixed_stream.py")
    assert cfg["prompt_sha256"] == hashlib.sha256((ROOT / "results/remediation_v3/latency_prompt.txt").read_text().encode()).hexdigest()
    assert cfg["teacher_token_sha256"] == hashlib.sha256(json.dumps(cfg["teacher_token_ids"]).encode()).hexdigest()
    assert len(cfg["teacher_token_ids"]) == cfg["decode"] + 1
    assert cfg["order"] == ["inclusive", "no_shadow", "no_shadow", "inclusive"]
    results = data["results"]
    assert list(results) == [f"{i}_{name}" for i, name in enumerate(cfg["order"])]
    arms = {name: [] for name in ("inclusive", "no_shadow")}
    for i, name in enumerate(cfg["order"]):
        trials = results[f"{i}_{name}"]["trials"]
        assert len(trials) == 2
        for trial in trials:
            assert len(trial["step_ms"]) == trial["decode_steps"] == cfg["decode"]
            assert trial["paid_writes"] >= 0
            arms[name].append(trial)
    pairs = []
    for left, right in ((0, 1), (3, 2)):
        for trial in range(2):
            inc = results[f"{left}_inclusive"]["trials"][trial]
            plain = results[f"{right}_no_shadow"]["trials"][trial]
            reduction = 1 - inc["paid_writes"] / plain["paid_writes"]
            pairs.append({"inclusive_trial": f"{left}:{trial}",
                          "no_shadow_trial": f"{right}:{trial}",
                          "inclusive_paid_writes": inc["paid_writes"],
                          "no_shadow_paid_writes": plain["paid_writes"],
                          "reduction_pct": 100 * reduction})
    summary = {}
    for name, trials in arms.items():
        steps = [step for trial in trials for step in trial["step_ms"]]
        summary[name] = {
            "paid_writes_per_50_step_run": [t["paid_writes"] for t in trials],
            "writes_saved_per_50_step_run": [t["writes_saved"] for t in trials],
            "itl_median_ms": float(np.median(steps)),
            "itl_p95_ms": float(np.quantile(steps, .95)),
            "ttft_median_ms": float(np.median([t["ttft_ms"] for t in trials])),
            "prefill_peak_max_gb": max(t["prefill_peak_memory_gb"] for t in trials),
            "decode_peak_max_gb": max(t["decode_peak_memory_gb"] for t in trials),
        }
    report = {"experiment": "SHADOW-FIXED-STREAM", "raw_sha256": sha(SOURCE),
              "pairs": pairs, "summary": summary,
              "write_gate": "pass" if all(p["reduction_pct"] >= 20 for p in pairs) else "fail",
              "limitations": "One real-text prompt, one Mistral model, teacher-forced 50-token stream and GPU-resident cold tier. Four ABBA arm runs with two measured trials each; this is a logical layer-token write comparison, not physical page traffic, measured energy, endurance or a general latency claim."}
    atomic_json(SOURCE.with_name("fixed_stream_report_20260929.json"), report)
    print(json.dumps({k: report[k] for k in ("write_gate", "pairs", "summary")}, indent=2))


if __name__ == "__main__":
    main()
