"""Trace the row-138 answer span through corrected TieredKV prefill only."""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from longbench_eval import TieredKVPolicy, generate_with_policy
from longbench_metrics import DATASET_TO_PROMPT
from provenance import atomic_json, runtime_identity


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/baseline_fidelity/icecache_tiered_row138/data/multifieldqa_en_e.jsonl"
OUT = ROOT / "results/baseline_fidelity/icecache_tiered_row138/five_order_span_trace.json"
MODEL = ROOT.parent / "models/mistral-7b-instruct"


def main():
    item = json.loads(DATA.read_text())
    prompt = "[INST] " + DATASET_TO_PROMPT["multifieldqa_en"].format(**item) + " [/INST]"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok = tokenizer(prompt, return_offsets_mapping=True, truncation=False)
    ids = tok.input_ids
    token_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    expected = "66f1b85caf8d6caef6ceff6cc224d1002cfb9c5cd3d24da2cf9af9ed7dc3545b"
    if token_hash != expected:
        raise RuntimeError("Prompt token IDs differ from paired evaluation")

    phrase = "Qaderi Silsila (Order), but also in the Chishti, Nakshbandi, Suharwardi, and Madaari Orders."
    start = prompt.index(phrase)
    end = start + len(phrase)
    evidence = [i for i, (a, b) in enumerate(tok.offset_mapping) if a < end and b > start]
    if not evidence:
        raise RuntimeError("Evidence span has no token offsets")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    policy = TieredKVPolicy(
        sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
        page_size=16, promote_top_pages=2, sttram_bifurcation_frac=.75,
        kv_dtype=torch.bfloat16)
    with torch.inference_mode():
        generate_with_policy(model, tokenizer, prompt, 1, policy, "cuda",
                             max_ctx=31500, task_name="multifieldqa_en")

    layers = []
    for layer in sorted(policy.layer_caches):
        cache = policy.layer_caches[layer]
        fast = set(cache.sram_pos)
        slow = set(cache.stt_pos)
        visible = set(policy._vis_stt_pos[layer])
        counts = {"fast": 0, "slow_visible": 0, "slow_hidden": 0, "dropped": 0}
        for pos in evidence:
            state = ("fast" if pos in fast else "slow_visible" if pos in visible
                     else "slow_hidden" if pos in slow else "dropped")
            counts[state] += 1
        layers.append({"layer": layer, **counts,
                       "fast_total": len(fast), "slow_total": len(slow),
                       "slow_visible_total": len(visible)})

    report = {
        "scope": "one-row post-prefill residency, before any decode step",
        "sample_id": hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest(),
        "input_tokens": len(ids), "input_token_sha256": token_hash,
        "phrase": phrase, "phrase_character_range": [start, end],
        "evidence_token_range": [evidence[0], evidence[-1] + 1],
        "evidence_token_count": len(evidence),
        "layers": layers,
        "runtime": runtime_identity(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(OUT, report)
    print(json.dumps({"evidence_tokens": len(evidence), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
