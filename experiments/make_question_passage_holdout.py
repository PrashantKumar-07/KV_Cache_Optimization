"""Freeze unseen documents for the answer-free passage-retrieval screen."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from provenance import atomic_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/baseline_fidelity"
OUT = BASE / "question_passage_holdout_20260930"
PRIOR = [BASE / "real_query_holdout_20260929/manifest.json",
         BASE / "real_query_holdout_20260929/batch2/manifest.json",
         BASE / "real_query_document_holdout_20260929/manifest.json"]
TASKS = ("multifieldqa_en", "qasper")
SEED = 20261002


def sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


def document_key(row):
    return sha(" ".join(row["context"].casefold().split()))


def main():
    target = OUT / "manifest.json"
    if target.exists():
        raise FileExistsError("Holdout manifest already frozen")
    prior = [json.loads(path.read_text()) for path in PRIOR]
    rng = random.Random(SEED)
    manifest = {
        "seed": SEED,
        "selection": "12 distinct previously unused contexts per task, one random row per context",
        "excluded_prior_manifest_sha256": [hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in PRIOR],
        "tasks": {},
    }
    for task in TASKS:
        source = (ROOT / prior[0]["tasks"][task]["source_file"]).resolve()
        rows = read_jsonl(source)
        excluded = set(range(25))
        for m in prior:
            excluded.update(m["tasks"][task]["selected_zero_based_indices"])
        if task == "multifieldqa_en":
            excluded.add(138)
        used = {document_key(rows[i]) for i in excluded}
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            key = document_key(row)
            if key not in used:
                groups[key].append(i)
        chosen_groups = sorted(rng.sample(sorted(groups), 12))
        indices = sorted(rng.choice(groups[key]) for key in chosen_groups)
        selected = [rows[i] for i in indices]
        dest = OUT / "data" / f"{task}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in selected))
        manifest["tasks"][task] = {
            "source_file": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_row_count": len(rows),
            "selected_zero_based_indices": indices,
            "selected_sample_ids": [sha(json.dumps(r, sort_keys=True)) for r in selected],
            "selected_context_keys": [document_key(r) for r in selected],
            "eligible_document_groups": len(groups),
            "slice_file": str(dest.relative_to(ROOT)),
            "slice_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
    atomic_json(target, manifest)
    print(json.dumps({task: manifest["tasks"][task]["selected_zero_based_indices"]
                      for task in TASKS}, indent=2))


if __name__ == "__main__":
    main()
