#!/usr/bin/env python3
"""ppl_eval.py — Sliding-window perplexity vs context length.

RETIRED from the paper's evidence (2026-09-16): this harness runs every
chunk under SDPA with no attention weights, so TieredKV's promotion path
(stt_attn-driven) never fires and H2O's decode scores freeze at their
prefill init -- it measures neutered policies. Recommissioning requires
the eager+output_attentions+update_scores treatment from latency_profile.py.
The old results/ppl_results.json (TieredKV at 1.5x attended tokens,
undisclosed) is archived, not cited. TieredKV config below is fixed for
that day (attention-matched expose=32, bf16)."""

Evaluates perplexity for all KV cache methods as input length increases.
This is the standard evaluation from StreamingLLM and Quest papers.

Usage:
    python experiments/ppl_eval.py \
        --model /path/to/mistral-7b-instruct \
        --output results/ppl_results.json \
        --max-len 32000 \
        --stride 512 \
        --n-docs 20
"""

import os, sys, json, argparse, math, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

_script_dir = os.path.dirname(os.path.abspath(__file__))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "longbench_eval", os.path.join(_script_dir, "longbench_eval.py"))
_lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lb)

FullCachePolicy    = _lb.FullCachePolicy
StreamingLLMPolicy = _lb.StreamingLLMPolicy
H2OPolicy          = _lb.H2OPolicy
SnapKVPolicy       = _lb.SnapKVPolicy
TieredKVPolicy     = _lb.TieredKVPolicy
to_tuple_kv        = _lb.to_tuple_kv
apply_tuple_kv_inplace = _lb.apply_tuple_kv_inplace
_attn_impl         = _lb._attn_impl


def _load_texts(hf_home, n_docs=20):
    """Load long text documents for PPL evaluation."""
    import glob as _glob
    
    # Try gov_report (we know it's cached)
    lb_dir = os.path.join(hf_home, "datasets", "downloads", "extracted")
    patterns = _glob.glob(os.path.join(lb_dir, "*", "data", "gov_report*.jsonl"))
    if patterns:
        texts = []
        for fp in patterns:
            with open(fp) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        ctx = obj.get("context") or obj.get("article") or ""
                        if len(ctx) > 2000:
                            texts.append(ctx)
                    except Exception:
                        pass
                    if len(texts) >= n_docs:
                        break
            if len(texts) >= n_docs:
                break
        if texts:
            print(f"  [ppl] Loaded {len(texts)} gov_report docs")
            return texts[:n_docs]

    # Fallback: synthetic long text
    base = ("The researchers investigated the properties of the proposed system "
            "using a comprehensive set of experiments and ablation studies. "
            "Results demonstrate significant improvements across all metrics. ") * 3000
    texts = [base[i:i+80000] for i in range(0, n_docs * 80000, 80000)][:n_docs]
    print(f"  [ppl] Using synthetic text ({len(texts)} chunks)")
    return texts


@torch.inference_mode()
def evaluate_ppl(model, tokenizer, texts, policy, budget, device,
                 max_len=32000, stride=512, name="method"):
    """Compute perplexity at log-spaced context length checkpoints."""
    checkpoints = [1024, 2048, 4096, 8192, 16384, 32000]
    checkpoints = [c for c in checkpoints if c <= max_len + stride]

    nll_sum = 0.0
    tok_count = 0
    ppl_at = {}

    for doc_idx, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc.input_ids[0]
        doc_len = input_ids.size(0)
        if doc_len < 512:
            continue

        past_kv = None
        seq_cursor = 0
        if hasattr(policy, "reset"):
            policy.reset()

        while seq_cursor < min(doc_len - 1, max_len):
            end = min(seq_cursor + stride, doc_len - 1, max_len)
            chunk  = input_ids[seq_cursor:end].unsqueeze(0).to(device)
            target = input_ids[seq_cursor + 1:end + 1].unsqueeze(0).to(device)
            pos    = torch.arange(seq_cursor, end, device=device).unsqueeze(0)

            with _attn_impl(model, "sdpa"):
                out = model(chunk, past_key_values=past_kv,
                            use_cache=True, position_ids=pos)

            # NLL on this chunk
            logits = out.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels  = target[:, :shift_logits.size(1)].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction="sum")
            nll_sum   += loss.item()
            tok_count += shift_labels.numel()

            # Apply eviction policy
            raw_past = out.past_key_values
            if raw_past is not None:
                tup = to_tuple_kv(raw_past)
                tup = policy(tup)
                apply_tuple_kv_inplace(raw_past, tup)
            past_kv = out.past_key_values
            seq_cursor = end

            # Record PPL at each checkpoint
            for ck in checkpoints:
                if seq_cursor >= ck and ck not in ppl_at:
                    ppl_at[ck] = math.exp(nll_sum / max(tok_count, 1))
                    print(f"    [{name}] ctx={ck:6d}  PPL={ppl_at[ck]:.3f}  "
                          f"doc {doc_idx+1}/{len(texts)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Fill remaining checkpoints with final PPL
    if tok_count > 0:
        final_ppl = math.exp(nll_sum / tok_count)
        for ck in checkpoints:
            if ck not in ppl_at:
                ppl_at[ck] = final_ppl

    return dict(sorted(ppl_at.items()))


def make_policy(name, budget, sink=4):
    recent = max(16, budget // 8)
    if name == "full":          return FullCachePolicy()
    if name == "streamingllm":  return StreamingLLMPolicy(start_size=sink, recent_size=budget - sink)
    if name == "h2o":           return H2OPolicy(budget=budget, start_size=sink, recent_size=recent)
    if name == "snapkv":        return SnapKVPolicy(budget=budget, start_size=sink, recent_size=recent)
    if name == "tieredkv":
        # Attention-matched setting: VRAM = budget (1024 tokens) like every
        # baseline, STT = budget as victim pool, expose quota 32 (was 512
        # with promote_top_pages=16, attending 1536 vs baselines' 1024 --
        # a 1.5x undisclosed advantage). bf16 residency like the baselines.
        import torch
        return TieredKVPolicy(sram_budget=budget, stt_budget=budget,
                              start_size=sink, recent_size=recent,
                              page_size=16, promote_top_pages=2,
                              stt_expose_quota=32,
                              kv_dtype=torch.bfloat16)
    raise ValueError(f"Unknown method: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--output",  default="results/ppl_results.json")
    parser.add_argument("--max-len", type=int, default=32000, dest="max_len")
    parser.add_argument("--stride",  type=int, default=512)
    parser.add_argument("--n-docs",  type=int, default=20,   dest="n_docs")
    parser.add_argument("--budget",  type=int, default=1024)
    parser.add_argument("--methods", nargs="+",
                        default=["full","streamingllm","h2o","snapkv","tieredkv"])
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True, attn_implementation="sdpa")
    model.eval()

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    texts = _load_texts(hf_home, n_docs=args.n_docs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = {}

    # Load existing results to enable resuming
    if os.path.exists(args.output):
        with open(args.output) as f:
            saved = json.load(f)
            results = saved.get("results", {})

    for mname in args.methods:
        if mname in results:
            print(f"  [SKIP] {mname} already done")
            continue
        print(f"\n{'='*50}\nMethod: {mname}\n{'='*50}")
        policy = make_policy(mname, args.budget)
        t0 = time.time()
        ppl_curve = evaluate_ppl(
            model, tokenizer, texts, policy, args.budget, args.device,
            max_len=args.max_len, stride=args.stride, name=mname)
        results[mname] = {"ppl_curve": ppl_curve, "elapsed_s": round(time.time()-t0, 1)}
        with open(args.output, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)
        print(f"  {mname}: {ppl_curve}  [saved]")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
