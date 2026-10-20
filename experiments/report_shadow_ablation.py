#!/usr/bin/env python3
"""Validate paired inclusive/no-shadow LongBench runs and summarize them."""
import argparse
import json
from pathlib import Path
from statistics import median
from provenance import atomic_json
from report_remediation import compare


def load_arm(root, name):
    single = root / f"{name}.json"
    if single.exists():
        return json.loads(single.read_text())
    short = json.loads((root / f"{name}_short.json").read_text())
    gov = json.loads((root / f"{name}_gov.json").read_text())
    varying = {"tasks", "max_samples", "dataset_sha256"}
    assert {k: v for k, v in short["config"].items() if k not in varying} == {
        k: v for k, v in gov["config"].items() if k not in varying}, "Short/Gov config mismatch"
    assert not set(short["results"]) & set(gov["results"])
    config = dict(short["config"])
    config["tasks"] = short["config"]["tasks"] + gov["config"]["tasks"]
    config["max_samples"] = {t: short["config"]["max_samples"] for t in short["results"]}
    config["max_samples"].update({t: gov["config"]["max_samples"] for t in gov["results"]})
    config["dataset_sha256"].update(gov["config"]["dataset_sha256"])
    return {"config": config, "results": {**short["results"], **gov["results"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="results/shadow_ablation_v2")
    args = parser.parse_args()
    root = Path(args.directory)
    inc = load_arm(root, "inclusive")
    plain = load_arm(root, "no_shadow")
    a, b = inc["config"], plain["config"]
    assert a["tiered_inclusive"] is True and b["tiered_inclusive"] is False
    assert {k: v for k, v in a.items() if k != "tiered_inclusive"} == {
        k: v for k, v in b.items() if k != "tiered_inclusive"}, "Arms differ beyond shadow policy"
    tasks = tuple(a["tasks"])
    assert set(tasks) == set(inc["results"]) == set(plain["results"])
    paired = {}
    totals = {name: {key: 0 for key in ("paid_writes", "writes_saved", "demoted", "promoted", "decode_steps")}
              for name in ("inclusive", "no_shadow")}
    for task in tasks:
        x, y = inc["results"][task]["tieredkv"], plain["results"][task]["tieredkv"]
        paired[task] = {"tieredkv": x, "no_shadow": y}
        for name, entry in (("inclusive", x), ("no_shadow", y)):
            for key in totals[name]:
                totals[name][key] += entry["cache_stats"][key]
    quality = compare(paired, "no_shadow", tasks=tasks)
    paid_delta = totals["no_shadow"]["paid_writes"] - totals["inclusive"]["paid_writes"]
    write_rates = {name: totals[name]["paid_writes"] / totals[name]["decode_steps"]
                   for name in totals}
    report = {
        "scope": "pilot; paired selected samples/tasks/model/seed, conditional bootstrap only",
        "runtime": a["runtime"], "tasks": tasks,
        "samples_per_task": {t: paired[t]["tieredkv"]["n"] for t in tasks},
        "quality": quality, "counters": totals,
        "paid_write_reduction": paid_delta,
        "paid_write_reduction_pct": 100 * paid_delta / totals["no_shadow"]["paid_writes"],
        "paid_writes_per_decode_step": write_rates,
        "paid_write_rate_reduction_pct": 100 * (1 - write_rates["inclusive"] / write_rates["no_shadow"]),
        "estimated_payload_reduction_bytes": paid_delta * 4096,
        "caveat": "Simulated layer-token counters; output lengths can differ between arms, "
                  "so compare paid writes per decode step as well as aggregate counts. "
                  "No physical STT writes or power measured. "
                  "Independent GPU timings are not a controlled latency comparison.",
    }
    latency_path = root / "latency.json"
    if latency_path.exists():
        latency = json.loads(latency_path.read_text())
        assert latency["config"]["runtime"] == a["runtime"], "Latency source mismatch"
        assert latency["config"]["order"] == ["inclusive", "no_shadow", "no_shadow", "inclusive"]
        blocks = latency["results"]
        report["latency"] = {
            "scope": "16k real-text fixed-length decode; same L40S GPU, ABBA blocks; prototype full loop",
            "block_itl_ms": {name: stats["itl_ms"] for name, stats in blocks.items()},
            "median_block_itl_ms": {
                arm: median(stats["itl_ms"] for name, stats in blocks.items() if name.endswith(arm))
                for arm in ("inclusive", "no_shadow")},
            "prefill_peak_gb": {name: stats["prefill_peak_memory_gb"]
                                for name, stats in blocks.items()},
            "decode_peak_gb": {name: stats["peak_memory_gb"]
                               for name, stats in blocks.items()},
        }
    atomic_json(root / "report.json", report)
    print(json.dumps({"quality_delta": quality["task_balanced_mean_difference"],
                      "quality_interval": quality["paired_bootstrap_95pct_interval"],
                      "paid_write_reduction": paid_delta}, indent=2))


if __name__ == "__main__":
    main()
