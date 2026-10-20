#!/usr/bin/env python3
"""Validate paired query-aware versus access-time TieredKV quality runs."""
import argparse
import json
from pathlib import Path
from provenance import atomic_json
from report_remediation import compare
from report_shadow_ablation import load_arm


def load_query(root):
    single = root / "query.json"
    if single.exists():
        return json.loads(single.read_text())
    short = json.loads((root / "query_short.json").read_text())
    gov = json.loads((root / "query_gov.json").read_text())
    variable = {"tasks", "max_samples", "dataset_sha256"}
    assert {k: v for k, v in short["config"].items() if k not in variable} == {
        k: v for k, v in gov["config"].items() if k not in variable}
    config = dict(short["config"])
    config["tasks"] = short["config"]["tasks"] + gov["config"]["tasks"]
    config["max_samples"] = {t: short["config"]["max_samples"] for t in short["results"]}
    config["max_samples"].update({t: gov["config"]["max_samples"] for t in gov["results"]})
    config["dataset_sha256"] = {**short["config"]["dataset_sha256"],
                                **gov["config"]["dataset_sha256"]}
    return {"config": config, "results": {**short["results"], **gov["results"]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", default="results/query_selection_pilot")
    p.add_argument("--control-directory", help="Use source-matched inclusive full run as LRU control")
    args = p.parse_args()
    root = Path(args.directory)
    query = load_query(root)
    lru = (load_arm(Path(args.control_directory), "inclusive")
           if args.control_directory else json.loads((root / "lru.json").read_text()))
    a, b = query["config"], lru["config"]
    assert a["strategy"] == "query"
    if args.control_directory:
        for key in ("model", "data_dir", "seed", "max_ctx", "runtime",
                    "tasks", "max_samples", "dataset_sha256"):
            assert a[key] == b[key], f"Control differs in {key}"
        assert b["tiered_inclusive"] is True
        assert b["stt_live_floor"] == 1 and b["write_aware_lambda"] == 0
        assert b["page_size"] == 16 and b["promote_top_pages"] == 2
        assert b["tiered_stt_expose"] in (None, 32) and a["stt_expose_quota"] == 32
        assert (a["tiered_vram_budget"], a["tiered_stt_budget"],
                a["recent"], a["sttram_bifurcation_frac"], a["kv_dtype"]) == (
                b["tiered_vram_budget"], b["tiered_stt_budget"], b["recent"],
                b["sttram_bifurcation_frac"], b["tiered_kv_dtype"])
    else:
        assert b["strategy"] == "lru"
        assert {k: v for k, v in a.items() if k != "strategy"} == {
            k: v for k, v in b.items() if k != "strategy"}, "Arms differ beyond selector"
    tasks = tuple(a["tasks"])
    assert set(tasks) == set(query["results"]) == set(lru["results"])
    paired = {t: {"tieredkv": query["results"][t]["tieredkv"],
                  "lru": lru["results"][t]["tieredkv"]} for t in tasks}
    quality = compare(paired, "lru", tasks=tasks)
    keys = ("paid_writes", "writes_saved", "promoted", "demoted", "decode_steps")
    counters = {arm: {key: sum(doc["results"][t]["tieredkv"]["cache_stats"][key]
                               for t in tasks) for key in keys}
                for arm, doc in (("query", query), ("lru", lru))}
    report = {"scope": "paired selected tasks/samples/model/seed; conditional bootstrap only",
              "runtime": a["runtime"], "selector_sha256": a["selector_sha256"],
              "control_directory": args.control_directory,
              "tasks": tasks, "quality": quality, "counters": counters,
              "caveat": "The same input samples can generate different output lengths. "
                        "Separate-GPU per-request elapsed times are not a controlled latency comparison. "
                        "STT tier is simulated on GPU."}
    atomic_json(root / "report.json", report)
    print(json.dumps({"quality_delta": quality["task_balanced_mean_difference"],
                      "quality_interval": quality["paired_bootstrap_95pct_interval"]}, indent=2))


if __name__ == "__main__":
    main()
