"""Reference-informed retention upper bound for one known failure; not a policy."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention import snapkv_importance
from longbench_eval import TieredKVPolicy, generate_with_policy
from longbench_metrics import DATASET_TO_PROMPT, DATASET_TO_METRIC
from provenance import atomic_json, runtime_identity
from tiered_kv_cache import TieredKVCache


DATA = ROOT / "results/baseline_fidelity/icecache_tiered_row138/data/multifieldqa_en_e.jsonl"
OUTDIR = ROOT / "results/baseline_fidelity/icecache_tiered_row138"
MODEL = ROOT.parent / "models/mistral-7b-instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-boost", action="store_true")
    args = parser.parse_args()
    item = json.loads(DATA.read_text())
    prompt = "[INST] " + DATASET_TO_PROMPT["multifieldqa_en"].format(**item) + " [/INST]"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok = tokenizer(prompt, return_offsets_mapping=True, truncation=False)
    digest = hashlib.sha256(json.dumps(tok.input_ids, separators=(",", ":")).encode()).hexdigest()
    if digest != "66f1b85caf8d6caef6ceff6cc224d1002cfb9c5cd3d24da2cf9af9ed7dc3545b":
        raise RuntimeError("Input token IDs differ from the frozen control")
    phrase = "Qaderi Silsila (Order), but also in the Chishti, Nakshbandi, Suharwardi, and Madaari Orders."
    start = prompt.index(phrase)
    end = start + len(phrase)
    evidence = [i for i, (a, b) in enumerate(tok.offset_mapping) if a < end and b > start]
    if len(evidence) != 35:
        raise RuntimeError("Evidence span differs from the prior trace")

    class OracleCache(TieredKVCache):
        def initial_bifurcation(self, k_all, v_all, q_all, importance_override=None):
            if importance_override is not None:
                raise RuntimeError("Unexpected external importance override")
            base = snapkv_importance(
                q_all[:, -self.cfg.window_size:], k_all, self.cfg.pool_kernel)
            boosted = base.clone()
            if not args.no_boost:
                boosted[evidence] = base.max() + 1
            return super().initial_bifurcation(
                k_all, v_all, q_all, importance_override=boosted)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()
    policy = TieredKVPolicy(
        sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
        page_size=16, promote_top_pages=2, sttram_bifurcation_frac=.75,
        kv_dtype=torch.bfloat16)
    policy.cache_cls = OracleCache
    with torch.inference_mode():
        prediction = generate_with_policy(
            model, tokenizer, prompt, 64, policy, "cuda",
            max_ctx=31500, task_name="multifieldqa_en")
    fast_by_layer = [sum(pos in policy.layer_caches[i].sram_pos for pos in evidence)
                     for i in sorted(policy.layer_caches)]
    if len(fast_by_layer) != 32 or (not args.no_boost and min(fast_by_layer) != 35):
        raise RuntimeError("Evidence-placement check failed")
    score = max(DATASET_TO_METRIC["multifieldqa_en"](
        prediction, answer, all_classes=item.get("all_classes", []))
        for answer in item["answers"]) * 100
    report = {
        "scope": "reference-informed single-row diagnostic; not deployable",
        "oracle_boost": not args.no_boost,
        "sample_id": hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest(),
        "input_tokens": len(tok.input_ids), "input_token_sha256": digest,
        "evidence_token_count": len(evidence),
        "evidence_fast_tokens_by_layer": fast_by_layer,
        "prediction": prediction, "score": score,
        "generated_tokens": policy.last_generation["generated_tokens"],
        "runtime": runtime_identity(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = OUTDIR / ("five_order_no_boost_control.json" if args.no_boost
                    else "five_order_oracle.json")
    atomic_json(out, report)
    print(json.dumps({k: report[k] for k in
                      ("input_tokens", "evidence_token_count", "generated_tokens",
                       "prediction", "score")}, indent=2))


if __name__ == "__main__":
    main()
