"""Compare fixed windows with sentence-aware windows on frozen unseen documents."""
import argparse
import bisect
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from longbench_metrics import DATASET_TO_PROMPT
from provenance import atomic_json, read_jsonl
from screen_question_lexical import terms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/baseline_fidelity/question_passage_holdout_20260930"
MODEL = ROOT.parent / "models/mistral-7b-instruct"
WIDTH = 128


def span_to_tokens(start, end, starts, ends, lo, hi):
    return (bisect.bisect_right(ends, start, lo, hi),
            bisect.bisect_left(starts, end, lo, hi))


def score_candidates(candidates, question):
    query = terms(question)
    df = Counter(word for _, _, words in candidates for word in words)
    n = len(candidates)
    return [sum(math.log((n + 1) / (df[w] + 1)) + 1
                for w in query & words) for _, _, words in candidates]


def fixed_window(prompt, offsets, lo, hi, question):
    if hi - lo <= WIDTH:
        return (lo, hi), 0.0
    starts = list(range(lo, hi - WIDTH + 1, 64))
    if starts[-1] != hi - WIDTH:
        starts.append(hi - WIDTH)
    candidates = [(i, i + WIDTH, terms(prompt[offsets[i][0]:offsets[i + WIDTH - 1][1]]))
                  for i in starts]
    scores = score_candidates(candidates, question)
    best = max(range(len(candidates)), key=lambda i: (scores[i], -candidates[i][0]))
    return candidates[best][:2], scores[best]


def sentence_window(prompt, context, offsets, lo, hi, ctx_start, question):
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]
    candidates = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", context):
        a, b = match.span()
        if not context[a:b].strip():
            continue
        tok_a, tok_b = span_to_tokens(ctx_start + a, ctx_start + b,
                                      starts, ends, lo, hi)
        if tok_b <= tok_a:
            continue
        candidates.append((tok_a, tok_b, terms(match.group())))
    if not candidates:
        return (lo, min(lo + WIDTH, hi)), 0.0
    scores = score_candidates(candidates, question)
    best = max(range(len(candidates)), key=lambda i: (scores[i], -candidates[i][0]))
    a, b, _ = candidates[best]
    # Keep the selected sentence centered in a fixed-size token window.
    center = (a + b) // 2
    begin = max(lo, min(center - WIDTH // 2, hi - WIDTH))
    return (begin, min(begin + WIDTH, hi)), scores[best]


def answer_spans(context, answers, ctx_start, starts, ends, lo, hi):
    spans = []
    for answer in answers:
        if len(answer) < 8 or answer.casefold() in {"yes", "no", "unanswerable"}:
            continue
        for match in re.finditer(re.escape(answer), context, flags=re.IGNORECASE):
            a, b = span_to_tokens(ctx_start + match.start(),
                                  ctx_start + match.end(), starts, ends, lo, hi)
            if 0 < b - a <= WIDTH:
                spans.append((a, b))
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    base = args.base.resolve()
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    report = {"scope": "exact-substring retrieval proxy, not answer quality",
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "window_tokens": WIDTH, "tasks": {}}
    selections = {"scope": "answer-free selected token ranges only",
                  "manifest_sha256": report["manifest_sha256"], "tasks": {}}
    for task in ("multifieldqa_en", "qasper"):
        meta = manifest["tasks"][task]
        path = ROOT / meta["slice_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["slice_sha256"]:
            raise RuntimeError("Frozen data slice changed")
        samples = []
        for row in read_jsonl(path):
            prompt = "[INST] " + DATASET_TO_PROMPT[task].format(**row) + " [/INST]"
            # Evaluation is about the original context, not model truncation.
            tok = tokenizer(prompt, return_offsets_mapping=True, truncation=False)
            offsets = tok.offset_mapping
            starts, ends = [a for a, _ in offsets], [b for _, b in offsets]
            ctx_start = prompt.index(row["context"])
            ctx_end = ctx_start + len(row["context"])
            positions = [i for i, (a, b) in enumerate(offsets)
                         if a >= ctx_start and b <= ctx_end and b > a]
            lo, hi = positions[0], positions[-1] + 1
            fixed, fixed_score = fixed_window(prompt, offsets, lo, hi, row["input"])
            passage, passage_score = sentence_window(
                prompt, row["context"], offsets, lo, hi, ctx_start, row["input"])
            gold = answer_spans(row["context"], row["answers"], ctx_start,
                                starts, ends, lo, hi)
            covers = lambda window: any(window[0] <= a and b <= window[1]
                                        for a, b in gold)
            samples.append({
                "sample_id": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                "prompt_tokens": len(tok.input_ids),
                "evaluable": bool(gold), "answer_occurrences": len(gold),
                "fixed_range": fixed, "passage_range": passage,
                "fixed_score": fixed_score, "passage_score": passage_score,
                "fixed_covers": covers(fixed) if gold else None,
                "passage_covers": covers(passage) if gold else None,
            })
        eligible = [x for x in samples if x["evaluable"]]
        report["tasks"][task] = {
            "rows": len(samples), "evaluable_rows": len(eligible),
            "fixed_hits": sum(x["fixed_covers"] for x in eligible),
            "passage_hits": sum(x["passage_covers"] for x in eligible),
            "new_losses": sum(x["fixed_covers"] and not x["passage_covers"]
                              for x in eligible),
            "samples": samples,
        }
        selections["tasks"][task] = [
            {"sample_id": x["sample_id"], "prompt_tokens": x["prompt_tokens"],
             "passage_range": x["passage_range"]} for x in samples]
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selections["runner_sha256"] = report["runner_sha256"]
    atomic_json(base / "selections.json", selections)
    atomic_json(base / "report.json", report)
    print(json.dumps({k: {m: v[m] for m in
                          ("rows", "evaluable_rows", "fixed_hits", "passage_hits", "new_losses")}
                      for k, v in report["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
