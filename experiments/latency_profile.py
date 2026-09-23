#!/usr/bin/env python3
"""latency_profile.py — Real GPU wall-clock latency measurement.

Measures TTFT (prefill) and ITL (decode) across context lengths and KV budgets.
TieredKV runs the headline equal-VRAM config (1024 VRAM + 2048 STT, bf16)
with promotion driven by real attention weights.

NOTE: the synthetic repeated-sentence prompt measures TIMING only, not
quality — uniform repeated KV makes every token interchangeable, so read
nothing about policy behavior into these numbers.
Outputs results/latency_headline.json (fresh file; the old
results/latency_profile.json was measured at the equal-total config with
promotion disabled and is retired to results/archive/).

Usage:
    python experiments/latency_profile.py \
        --model /path/to/mistral-7b-instruct \
        --output results/latency_headline.json
"""

import os, sys, json, argparse, time
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


def cuda_sync_time(fn):
    """Time a GPU operation with proper synchronization."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, result


def make_policy(name, budget, sink=4):
    recent = max(16, budget // 8)
    vram   = budget // 3
    stt    = budget - vram
    if name == "full":
        return FullCachePolicy()
    if name == "streamingllm":
        return StreamingLLMPolicy(start_size=sink, recent_size=budget - sink)
    if name == "h2o":
        return H2OPolicy(budget=budget, start_size=sink, recent_size=recent)
    if name == "snapkv":
        return SnapKVPolicy(budget=budget, start_size=sink, recent_size=recent)
    if name == "tieredkv":
        # Headline equal-VRAM config (matches results/final_equal_vram.json):
        # 1024 VRAM + 2048 STT, bf16 residency. Promotion needs real
        # attention weights (see decode loop) — without them the STT side
        # stays empty, nothing ever promotes, and you'd benchmark a
        # neutered policy while still paying its bookkeeping.
        return TieredKVPolicy(sram_budget=1024, stt_budget=2048,
                              start_size=sink, recent_size=recent,
                              page_size=16, promote_top_pages=2,
                              kv_dtype=torch.bfloat16)
    raise ValueError(f"Unknown method: {name}")


@torch.inference_mode()
def profile_one(model, tokenizer, policy, ctx_len, n_decode=50, n_warmup=2,
                n_repeat=3, device="cuda"):
    """Measure TTFT and ITL for a given context length and policy.
    
    Returns:
        ttft_ms (float): median prefill time in ms
        itl_ms  (float): median per-token decode time in ms
        memory_gb (float): peak VRAM used during decode
    """
    # Build a synthetic prompt of exactly ctx_len tokens
    # Use repetition of a real sentence so tokenization is natural
    filler = "The quick brown fox jumps over the lazy dog . " * 3000
    enc = tokenizer(filler, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids[0, :ctx_len].unsqueeze(0).to(device)

    ttft_list, itl_list = [], []

    for trial in range(n_warmup + n_repeat):
        if hasattr(policy, "reset"):
            policy.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        pos = torch.arange(ctx_len, device=device).unsqueeze(0)

        # TTFT: prefill
        with _attn_impl(model, "sdpa"):
            ttft_ms, out = cuda_sync_time(
                lambda: model(input_ids, use_cache=True, position_ids=pos)
            )

        raw_past = out.past_key_values
        tup = to_tuple_kv(raw_past)
        tup = policy(tup)
        apply_tuple_kv_inplace(raw_past, tup)
        past_kv = raw_past
        next_tok = out.logits[:, -1:].argmax(dim=-1)

        # ITL: decode. output_attentions=True is load-bearing: H2O and
        # TieredKV score eviction/promotion off the real attention weights
        # each step. With False the STT side stays empty, promotion never
        # fires, and the profile measures a neutered policy at full
        # bookkeeping cost — worst of both worlds.
        step_times = []
        for step in range(n_decode):
            true_pos = torch.tensor([[ctx_len + step]], device=device)
            itl_step_ms, out = cuda_sync_time(
                lambda: model(next_tok, past_key_values=past_kv,
                              use_cache=True, position_ids=true_pos,
                              output_attentions=True)
            )
            step_times.append(itl_step_ms)

            raw_past = out.past_key_values
            if hasattr(policy, "update_scores") and out.attentions is not None:
                try:
                    policy.update_scores(raw_past, list(out.attentions))
                except Exception:
                    pass
            tup = to_tuple_kv(raw_past)
            tup = policy(tup)
            apply_tuple_kv_inplace(raw_past, tup)
            past_kv = raw_past
            next_tok = out.logits[:, -1:].argmax(dim=-1)

        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                       if torch.cuda.is_available() else 0.0)

        if trial >= n_warmup:
            ttft_list.append(ttft_ms)
            itl_list.append(np.median(step_times))

    return {
        "ttft_ms": round(float(np.median(ttft_list)), 2),
        "itl_ms":  round(float(np.median(itl_list)), 3),
        "peak_memory_gb": round(peak_mem_gb, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--output",   default="results/latency_headline.json")
    parser.add_argument("--ctx-lens", nargs="+", type=int, dest="ctx_lens",
                        default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--budget",   type=int, default=1024)
    parser.add_argument("--n-decode", type=int, default=50, dest="n_decode")
    parser.add_argument("--n-repeat", type=int, default=3,  dest="n_repeat")
    parser.add_argument("--n-warmup", type=int, default=1,  dest="n_warmup")
    parser.add_argument("--methods",  nargs="+",
                        default=["full","streamingllm","h2o","snapkv","tieredkv"])
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            saved = json.load(f)
            results = saved.get("results", {})

    for mname in args.methods:
        if mname not in results:
            results[mname] = {}

        for ctx in args.ctx_lens:
            key = str(ctx)
            if key in results[mname]:
                print(f"  [SKIP] {mname} ctx={ctx}")
                continue

            print(f"\n[{mname}] ctx_len={ctx}...")
            policy = make_policy(mname, args.budget)
            try:
                stats = profile_one(
                    model, tokenizer, policy, ctx,
                    n_decode=args.n_decode,
                    n_warmup=args.n_warmup,
                    n_repeat=args.n_repeat,
                    device=args.device,
                )
                results[mname][key] = stats
                print(f"  TTFT={stats['ttft_ms']:.1f}ms  "
                      f"ITL={stats['itl_ms']:.2f}ms/tok  "
                      f"Mem={stats['peak_memory_gb']:.2f}GB")
            except torch.cuda.OutOfMemoryError:
                results[mname][key] = {"ttft_ms": None, "itl_ms": None,
                                       "peak_memory_gb": None, "oom": True}
                print(f"  OOM at ctx={ctx}")

            with open(args.output, "w") as f:
                json.dump({"config": vars(args), "results": results}, f, indent=2)
            print(f"  [saved → {args.output}]")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
