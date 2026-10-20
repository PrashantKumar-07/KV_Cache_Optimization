#!/usr/bin/env python3
"""Validate and summarize the paired bf16/FP8 cold-tier pilot."""
import json
from pathlib import Path
import numpy as np
from provenance import atomic_json


def main():
    root = Path("results/fp8_cold_pilot")
    docs = {name: json.loads((root / f"{name}_5.json").read_text())
            for name in ("bf16", "fp8")}
    a, b = docs["bf16"], docs["fp8"]
    variable = {"tiered_stt_kv_dtype"}
    assert {k: v for k, v in a["config"].items() if k not in variable} == {
        k: v for k, v in b["config"].items() if k not in variable}
    assert a["config"]["tiered_stt_kv_dtype"] == "same"
    assert b["config"]["tiered_stt_kv_dtype"] == "float8_e4m3fn"
    ar = a["results"]["qasper"]["tieredkv"]
    br = b["results"]["qasper"]["tieredkv"]
    assert ar["n"] == br["n"] == 5
    assert [x["sample_id"] for x in ar["samples"]] == [x["sample_id"] for x in br["samples"]]
    diff = np.array([y["score"] - x["score"] for x, y in zip(ar["samples"], br["samples"])])
    rng = np.random.default_rng(0)
    draws = rng.choice(diff, (10000, len(diff)), replace=True).mean(1)
    model = json.loads((Path(a["config"]["model"]) / "config.json").read_text())
    elements = (a["config"]["tiered_stt_budget"] * model["num_hidden_layers"] *
                model["num_key_value_heads"] * (model["hidden_size"] // model["num_attention_heads"]) * 2)
    report = {
        "scope": "Five fixed Qasper examples; FP8 minus bf16; same source/runtime/config except cold dtype",
        "scores": {"bf16": ar["score"], "fp8": br["score"]},
        "mean_difference": float(diff.mean()),
        "paired_bootstrap_95pct_interval": np.quantile(draws, [.025, .975]).tolist(),
        "identical_predictions": sum(x["prediction"] == y["prediction"]
                                     for x, y in zip(ar["samples"], br["samples"])),
        "cold_capacity_bytes": {"bf16": elements * 2, "fp8": elements},
        "limitations": "Selected five-example pilot; no task/model/seed generality, latency, or peak-memory claim."
    }
    atomic_json(root / "report_5.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
