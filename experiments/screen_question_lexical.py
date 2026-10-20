"""Answer-free question/window overlap screen on one known failure."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from longbench_metrics import DATASET_TO_PROMPT
from provenance import atomic_json


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/baseline_fidelity/icecache_tiered_row138"
MODEL = ROOT.parent / "models/mistral-7b-instruct"
STOP = {"a", "an", "and", "are", "did", "do", "does", "e", "for", "from",
        "in", "is", "of", "on", "the", "to", "was", "were", "what", "which", "who"}


def terms(text):
    out = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOP or len(word) < 3:
            continue
        if len(word) > 5 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        if len(word) > 5 and word.endswith("e"):
            word = word[:-1]
        out.add(word)
    return out


def main():
    item = json.loads((BASE / "data/multifieldqa_en_e.jsonl").read_text())
    prompt = "[INST] " + DATASET_TO_PROMPT["multifieldqa_en"].format(**item) + " [/INST]"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok = tokenizer(prompt, return_offsets_mapping=True, truncation=False)
    digest = hashlib.sha256(json.dumps(tok.input_ids, separators=(",", ":")).encode()).hexdigest()
    if digest != "66f1b85caf8d6caef6ceff6cc224d1002cfb9c5cd3d24da2cf9af9ed7dc3545b":
        raise RuntimeError("Prompt differs from paired case")
    ctx_start = prompt.index(item["context"])
    ctx_end = ctx_start + len(item["context"])
    context_tokens = [i for i, (a, b) in enumerate(tok.offset_mapping)
                      if a >= ctx_start and b <= ctx_end and b > a]
    lo, hi = context_tokens[0], context_tokens[-1] + 1
    query = terms(item["input"])
    width, stride = 128, 64
    starts = list(range(lo, hi - width + 1, stride))
    if starts[-1] != hi - width:
        starts.append(hi - width)
    windows = []
    for start in starts:
        char_start = tok.offset_mapping[start][0]
        char_end = tok.offset_mapping[start + width - 1][1]
        windows.append((start, start + width, terms(prompt[char_start:char_end])))
    df = Counter(term for _, _, words in windows for term in words)
    scores = [sum(math.log((len(windows) + 1) / (df[t] + 1)) + 1
                  for t in query & words) for _, _, words in windows]
    best = max(range(len(windows)), key=lambda i: (scores[i], -windows[i][0]))
    chosen = windows[best]

    # Evaluation labels enter only after the answer-free ranking is fixed.
    trace = json.loads((BASE / "five_order_span_trace.json").read_text())
    ev_lo, ev_hi = trace["evidence_token_range"]
    covered = max(0, min(chosen[1], ev_hi) - max(chosen[0], ev_lo))
    report = {
        "scope": "answer-free lexical ranking; evidence used only after selection",
        "sample_id": trace["sample_id"], "input_token_sha256": digest,
        "question_terms": sorted(query), "context_token_range": [lo, hi],
        "window_tokens": width, "stride_tokens": stride,
        "num_windows": len(windows), "chosen_token_range": list(chosen[:2]),
        "chosen_question_terms": sorted(query & chosen[2]),
        "chosen_score": scores[best],
        "evidence_token_range": [ev_lo, ev_hi],
        "evidence_tokens_covered": covered,
        "evidence_tokens_total": ev_hi - ev_lo,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(BASE / "question_lexical_screen.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
