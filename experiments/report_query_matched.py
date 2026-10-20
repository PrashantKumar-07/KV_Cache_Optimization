#!/usr/bin/env python3
"""Validate query-aware TieredKV against source-matched in-harness SnapKV."""
import argparse
import hashlib
import json
from pathlib import Path
from provenance import atomic_json
from report_remediation import compare


def merge(root, stem):
    short = json.loads((root / f"{stem}_short.json").read_text())
    gov = json.loads((root / f"{stem}_gov.json").read_text())
    variable = {"tasks", "max_samples", "dataset_sha256"}
    assert {k: v for k, v in short["config"].items() if k not in variable} == {
        k: v for k, v in gov["config"].items() if k not in variable}
    assert not set(short["results"]) & set(gov["results"])
    c = dict(short["config"])
    c["tasks"] = short["config"]["tasks"] + gov["config"]["tasks"]
    c["max_samples"] = {t: short["config"]["max_samples"] for t in short["results"]}
    c["max_samples"].update({t: gov["config"]["max_samples"] for t in gov["results"]})
    c["dataset_sha256"] = {**short["config"]["dataset_sha256"],
                           **gov["config"]["dataset_sha256"]}
    return c, {**short["results"], **gov["results"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", default="results/query_selection_matched")
    p.add_argument("--snap-directory", default=None)
    p.add_argument("--vram", type=int, default=992)
    p.add_argument("--stt", type=int, default=2048)
    args = p.parse_args()
    root = Path(args.directory)
    query_config, query = merge(root, "query")
    snap_config, snap = merge(Path(args.snap_directory) if args.snap_directory else root, "snap")
    assert query_config["driver_sha256"] == hashlib.sha256(
        Path(__file__).with_name("evaluate_query_budget.py").read_bytes()).hexdigest()
    assert query_config["selector_sha256"] == hashlib.sha256(
        Path(__file__).with_name("evaluate_query_selection.py").read_bytes()).hexdigest()
    for key in ("model", "data_dir", "seed", "max_ctx", "runtime",
                "max_samples", "dataset_sha256"):
        assert query_config[key] == snap_config[key], f"Mismatch in {key}"
    assert set(query_config["tasks"]) == set(snap_config["tasks"])
    assert query_config["strategy"] == "query"
    assert query_config["tiered_vram_budget"] == args.vram
    assert query_config["tiered_stt_budget"] == args.stt
    assert query_config["stt_expose_quota"] == 32
    assert snap_config["budget"] == 1024 and snap_config["methods"] == ["snapkv"]
    assert query_config["recent"] == snap_config["recent"] == 128
    tasks = tuple(query_config["tasks"])
    assert set(tasks) == set(query) == set(snap)
    paired = {t: {"tieredkv": query[t]["tieredkv"],
                  "snapkv": snap[t]["snapkv"]} for t in tasks}
    quality = compare(paired, "snapkv", tasks=tasks)
    assert (args.vram + min(32, args.stt) == snap_config["budget"]
            or args.vram + args.stt == snap_config["budget"])
    report = {"scope": (f"query {args.vram} fast + {args.stt} slow logical slots, "
                         f"maximum attended {args.vram + min(32, args.stt)}; "
                         "SnapKV 1024 slots; same source/datasets/samples/model/seed"),
              "runtime": query_config["runtime"],
              "selector_sha256": query_config["selector_sha256"],
              "driver_sha256": query_config["driver_sha256"],
              "quality": quality,
              "caveat": "SnapKV is the in-harness approximation, not upstream-faithful. "
                        "TieredKV's slow tier is simulated on GPU. "
                        "The arms may have different maximum attended widths. "
                        "Interval is conditional on sampled tasks/examples/model/seed."}
    atomic_json(root / "report.json", report)
    print(json.dumps({"quality_delta": quality["task_balanced_mean_difference"],
                      "quality_interval": quality["paired_bootstrap_95pct_interval"]}, indent=2))


if __name__ == "__main__":
    main()
