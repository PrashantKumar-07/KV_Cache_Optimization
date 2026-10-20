"""Freeze one question per unseen LongBench document for paired evaluation."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from provenance import atomic_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/baseline_fidelity/real_query_holdout_20260929"
OUT = ROOT / "results/baseline_fidelity/real_query_document_holdout_20260929"
SEED = 20261001


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    return " ".join(value.casefold().split())


def main():
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Frozen manifest already exists")
    old = [json.loads((folder / "manifest.json").read_text())
           for folder in (OLD, OLD / "batch2")]
    rng = random.Random(SEED)
    manifest = {
        "selection_seed": SEED,
        "selection_rule": "Four distinct source contexts per task, one row per context; exclude raw and casefold/whitespace-normalized contexts in the earlier development prefix and two prior holdouts; sample groups and row within group with Python Random seed",
        "prior_manifest_sha256": [hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
                                  for folder in (OLD, OLD / "batch2")],
        "tasks": {},
    }
    for task, previous in old[0]["tasks"].items():
        source = ROOT / previous["source_file"]
        rows = read_jsonl(source)
        excluded = set(range(previous["excluded_prior_prefix_count"]))
        for tranche in old:
            excluded.update(tranche["tasks"][task]["selected_zero_based_indices"])
        used = {sha(normalized(rows[i]["context"])) for i in excluded}
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            key = sha(normalized(row["context"]))
            if key not in used:
                groups[key].append(i)
        if len(groups) < 4:
            raise RuntimeError(f"{task}: only {len(groups)} clean document groups")
        chosen_groups = sorted(rng.sample(sorted(groups), 4))
        indices = sorted(rng.choice(groups[group]) for group in chosen_groups)
        chosen_rows = [rows[i] for i in indices]
        dest = OUT / "data" / f"{task}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen_rows))
        manifest["tasks"][task] = {
            "source_file": previous["source_file"],
            "source_sha256": previous["source_sha256"],
            "source_row_count": len(rows),
            "excluded_prior_prefix_count": previous["excluded_prior_prefix_count"],
            "selected_zero_based_indices": indices,
            "selected_sample_ids": [sha(json.dumps(r, sort_keys=True)) for r in chosen_rows],
            "selected_context_sha256": [sha(r["context"]) for r in chosen_rows],
            "selected_normalized_context_sha256": [sha(normalized(r["context"])) for r in chosen_rows],
            "eligible_clean_document_groups": len(groups),
            "slice_file": str(dest.relative_to(ROOT)),
            "slice_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
    atomic_json(OUT / "manifest.json", manifest)
    print(json.dumps({"manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
                      "selected_indices": {t: m["selected_zero_based_indices"]
                                           for t, m in manifest["tasks"].items()},
                      "eligible_groups": {t: m["eligible_clean_document_groups"]
                                          for t, m in manifest["tasks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
