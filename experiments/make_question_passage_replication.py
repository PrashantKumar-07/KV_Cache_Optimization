"""Freeze a second document-disjoint set without changing passage selection."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from provenance import atomic_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/baseline_fidelity"
OUT = BASE / "question_passage_replication_20260930"
PRIOR = [BASE / "real_query_holdout_20260929/manifest.json",
         BASE / "real_query_holdout_20260929/batch2/manifest.json",
         BASE / "real_query_document_holdout_20260929/manifest.json",
         BASE / "question_passage_holdout_20260930/manifest.json"]
TASKS = ("multifieldqa_en", "qasper")


def sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


def key(row):
    return sha(" ".join(row["context"].casefold().split()))


def main():
    target = OUT / "manifest.json"
    if target.exists():
        raise FileExistsError("Replication set is already frozen")
    old = [json.loads(path.read_text()) for path in PRIOR]
    rng = random.Random(20261003)
    manifest = {
        "seed": 20261003,
        "selection": "12 previously unseen document groups per task, one row per group",
        "excluded_prior_manifest_sha256": [hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in PRIOR],
        "tasks": {},
    }
    for task in TASKS:
        src = Path(old[-1]["tasks"][task]["source_file"])
        rows = read_jsonl(src)
        excluded = set(range(25))
        for prior in old:
            excluded.update(prior["tasks"][task]["selected_zero_based_indices"])
        if task == "multifieldqa_en":
            excluded.add(138)
        excluded_keys = {key(rows[i]) for i in excluded}
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            identity = key(row)
            if identity not in excluded_keys:
                groups[identity].append(i)
        picked_groups = sorted(rng.sample(sorted(groups), 12))
        indices = sorted(rng.choice(groups[g]) for g in picked_groups)
        selected = [rows[i] for i in indices]
        dest = OUT / "data" / f"{task}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in selected))
        manifest["tasks"][task] = {
            "source_file": str(src),
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "source_row_count": len(rows),
            "selected_zero_based_indices": indices,
            "selected_sample_ids": [sha(json.dumps(r, sort_keys=True)) for r in selected],
            "selected_context_keys": [key(r) for r in selected],
            "eligible_document_groups": len(groups),
            "slice_file": str(dest.relative_to(ROOT)),
            "slice_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
    manifest["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(target, manifest)
    print(json.dumps({t: manifest["tasks"][t]["selected_zero_based_indices"]
                      for t in TASKS}, indent=2))


if __name__ == "__main__":
    main()
