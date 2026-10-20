"""Check whether released IceCache LongBench inputs match our evaluation.

Run with PYTHONPATH=src:experiments and an external IceCache checkout.
This checks input fidelity only; it does not claim model or kernel parity.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from datasets import load_from_disk
from longbench_eval import DATASET_TO_MAXGEN, DATASET_TO_PROMPT, task_data_path
from provenance import atomic_json, read_jsonl, runtime_identity


TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "gov_report", "triviaqa")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--icecache-root", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.icecache_root.resolve()
    benchmark = root / "IceCache/benchmark"
    prompt_path = benchmark / "longbench_config/dataset2prompt.json"
    maxgen_path = benchmark / "longbench_config/dataset2maxlen.json"
    prompt = json.loads(prompt_path.read_text())
    maxgen = json.loads(maxgen_path.read_text())
    rows = {}
    for task in TASKS:
        upstream = list(load_from_disk(str(benchmark / "LongBench" / task)))
        standard_path = args.data_dir / f"{task}.jsonl"
        selected_path = task_data_path(args.data_dir, task)
        standard = read_jsonl(standard_path)
        selected = read_jsonl(selected_path)
        rows[task] = {
            "upstream_rows": len(upstream),
            "local_standard_rows": len(standard),
            "standard_rows_equal": upstream == standard,
            "selected_control_file": selected_path.name,
            "selected_control_rows": len(selected),
            "matches_selected_control": upstream == selected,
            "prompt_equal": prompt[task] == DATASET_TO_PROMPT[task],
            "max_generation_equal": maxgen[task] == DATASET_TO_MAXGEN[task],
            "local_standard_sha256": sha(standard_path),
            "selected_control_sha256": sha(selected_path),
        }
    report = {
        "scope": "dataset rows, task prompt templates, and generation limits; not model/kernel parity",
        "icecache_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "icecache_prompt_sha256": sha(prompt_path),
        "icecache_maxgen_sha256": sha(maxgen_path),
        "local_eval_sha256": sha(Path(__file__).with_name("longbench_eval.py")),
        "checker_sha256": sha(Path(__file__)),
        "runtime": runtime_identity(),
        "tasks": rows,
    }
    atomic_json(args.output, report)
    print(json.dumps({task: {k: v for k, v in item.items()
                             if not k.endswith("sha256")}
                      for task, item in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
