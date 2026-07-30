"""
model_wrapper.py  (Option A)
----------------------------
Hook the 3-tier KV cache into a REAL Transformer (Llama-3-8B) and measure the
things that DO NOT transfer from synthetic data: real per-layer accuracy, real
migration volume, real occupancy -- one `TieredKVCache` per layer.

HOW WE CAPTURE REAL K/V (robust, version-independent)
-----------------------------------------------------
HuggingFace Llama calls `torch.nn.functional.scaled_dot_product_attention`
exactly once per decoder layer. By the time SDPA is called:
  * RoPE has already been applied to Q and K, and
  * GQA key/value heads have already been expanded (`repeat_kv`) to the full
    number of query heads (32 for Llama-3-8B).
So if we wrap SDPA for the duration of ONE forward pass, we record, in layer
order, the exact post-RoPE / post-repeat (Q, K, V) that attention actually uses
-- shape (batch, 32, seq, 128). No monkeypatching of the generation loop, no
brittle access to model internals; just a stable PyTorch API.

GQA NOTE: capturing post-`repeat_kv` means we route at QUERY-HEAD granularity
(H=32). This gives EXACT attention output and accuracy. A GQA-aware deployment
could instead share one cache per 8-head KV group (up to 4x less KV memory) --
that is a memory optimization, not an accuracy change, and is left as future
work. The accuracy / recall claim this POC makes is unaffected.

WHAT THIS PRODUCES
------------------
The same measurement machinery as the synthetic experiments (Full oracle vs
TieredKV, cosine accuracy, GOPs, migration, latency/energy) but fed with REAL
captured tensors -- per layer and aggregated. This is the paper-ready run.

Local smoke test (no model download, no transformers needed):
    python experiments/model_wrapper.py --smoke
Server run (real weights):
    python experiments/model_wrapper.py --model meta-llama/Meta-Llama-3-8B \
        --text-file prompt.txt --prompt-len 256 --sram 128 --stt 256 \
        --layers 0 1 2 3 --json results/llama_layers.json
"""

import os, sys, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from tiered_kv_cache import TieredKVCache, TieredConfig
from baselines import FullAttention
from cost_model import CostModel

# ---------------------------------------------------------------- capture
class SDPACapture:
    """Context manager that records every scaled_dot_product_attention call.

    Patches torch.nn.functional.scaled_dot_product_attention. On each call it
    stores (q, k, v) detached-and-cloned, then delegates to the real SDPA so the
    model still runs normally. Calls arrive in layer order within a forward pass.
    """
    def __init__(self):
        self.calls = []          # list of (q, k, v) tensors, layer order
        self._orig = None

    def __enter__(self):
        self._orig = F.scaled_dot_product_attention

        def patched(query, key, value, *args, **kwargs):
            # query/key/value: (batch, n_heads, seq, head_dim), post-RoPE, post-repeat_kv
            self.calls.append((
                query.detach().float().cpu().clone(),
                key.detach().float().cpu().clone(),
                value.detach().float().cpu().clone(),
            ))
            return self._orig(query, key, value, *args, **kwargs)

        F.scaled_dot_product_attention = patched
        return self

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self._orig
        return False


def capture_layers(model, input_ids):
    """Run one prefill forward and return per-layer (q, k, v) at (H, seq, D).

    We take batch index 0 and squeeze it out, giving the (num_heads, seq,
    head_dim) convention the cache expects. Returns a list indexed by layer.
    """
    cap = SDPACapture()
    with torch.no_grad(), cap:
        model(input_ids=input_ids, use_cache=False)
    layers = []
    for (q, k, v) in cap.calls:
        # (1, H, S, D) -> (H, S, D); if batch>1 we keep row 0 only
        layers.append((q[0], k[0], v[0]))
    return layers


# ---------------------------------------------------------------- layer sources
def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def synthetic_layers(num_layers, H, D, seq, anchor_frac=0.35, scale=3.0, seed=42):
    """Fabricate per-layer (q, k, v) with the SAME long-range structure the
    synthetic experiments use, so `--smoke` exercises the real code path with
    NO transformers / model download.

    Returns (layers, masks):
      layers : list of (q, k, v) each (H, seq, D)
      masks  : list of per-DECODE-step bool lists (True == long-range anchor step),
               so accuracy can be split anchor/local exactly like compare_accuracy.
    The prompt/decode split is applied later by the evaluator; anchors are injected
    into the tail (decode) queries and steered at a random early key.
    """
    g = torch.Generator().manual_seed(seed)
    layers, masks = [], []
    # decode tail is whatever sits past prompt-len; we mark anchors across the
    # whole sequence and the evaluator keeps only the decode-region flags.
    for L in range(num_layers):
        k = torch.randn(H, seq, D, generator=g)
        v = torch.randn(H, seq, D, generator=g)
        q = torch.randn(H, seq, D, generator=g)
        flags = [False] * seq
        for t in range(seq):
            if torch.rand(1, generator=g).item() < anchor_frac:
                anchor = torch.randint(low=4, high=max(5, seq // 4),
                                       size=(1,), generator=g).item()
                q[:, t, :] = scale * k[:, anchor, :] + 0.3 * torch.randn(H, D, generator=g)
                flags[t] = True
        layers.append((q, k, v))
        masks.append(flags)
    return layers, masks


def real_layers(model_name, text, prompt_len, device="cpu"):
    """Load a real HF causal LM, capture post-RoPE/post-repeat_kv (q,k,v) per
    layer for one prefill of `text` (truncated to prompt_len tokens).

    Imported lazily so the smoke path needs neither transformers nor weights.
    Returns (layers, masks) with masks=None (no ground-truth anchor labels).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()

    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=prompt_len).input_ids.to(device)
    layers = capture_layers(model, ids)
    return layers, None


# ---------------------------------------------------------------- per-layer eval
def eval_layer(qkv, split, sram, stt, flags=None):
    """Replay ONE layer through Full oracle + TieredKV.

    qkv   : (q, k, v) each (H, seq, D) -- exact post-RoPE / post-repeat tensors.
    split : first `split` tokens are the initial prompt; the rest are streamed
            one-at-a-time as a decode replay (same shape as compare_accuracy).
    flags : optional per-position anchor labels (synthetic only) for the
            anchor/local accuracy breakdown; None for real captures.

    Returns a dict of measured numbers for this layer plus the tiered metrics.
    """
    q_all, k_all, v_all = qkv
    H, seq, D = k_all.shape
    split = max(1, min(split, seq - 1))

    k_p, v_p, q_p = k_all[:, :split, :], v_all[:, :split, :], q_all[:, :split, :]

    oracle = FullAttention()
    oracle.reset_prompt(k_p, v_p, q_p)

    tiered = TieredKVCache(TieredConfig(
        num_heads=H, head_dim=D, sink_size=4, window_size=16,
        sram_capacity=sram, sttram_capacity=stt, page_size=16,
        promote_top_pages=2, store_dram=False))
    tiered.initial_bifurcation(k_p, v_p, q_p)

    sims, anchor_sims, local_sims = [], [], []
    pos = split
    for t in range(split, seq):
        q = q_all[:, t:t + 1, :]
        nk = k_all[:, t:t + 1, :]
        nv = v_all[:, t:t + 1, :]
        o_out = oracle.step(q, nk, nv, pos)
        t_out = tiered.step(q, nk, nv, pos)
        s = cos(t_out, o_out)
        sims.append(s)
        if flags is not None:
            (anchor_sims if flags[t] else local_sims).append(s)
        pos += 1

    m = tiered.metrics
    acc_all = sum(sims) / len(sims) if sims else float("nan")
    acc_anchor = (sum(anchor_sims) / len(anchor_sims)) if anchor_sims else float("nan")
    acc_local = (sum(local_sims) / len(local_sims)) if local_sims else float("nan")
    return {
        "H": H, "D": D, "seq": seq, "prompt": split, "decode_steps": seq - split,
        "acc_all": acc_all, "acc_anchor": acc_anchor, "acc_local": acc_local,
        "tiered_GOPs": m.total_gops,
        "oracle_GOPs": oracle.total_macs * 2 / 1e9,
        "promoted": m.total_promoted, "demoted": m.total_demoted,
        "dropped": m.total_dropped,
        "peak_sram_tokens": m.peak_sram_tokens,
        "peak_sttram_tokens": m.peak_sttram_tokens,
        "peak_sram_KB": round(m.kv_bytes(m.peak_sram_tokens) / 1024, 3),
        "latency_us": m.total_latency_us, "energy_nj": m.total_energy_nj,
    }, m


# ---------------------------------------------------------------- reporting
def _fmt(x):
    return "  nan" if (isinstance(x, float) and x != x) else f"{x:.4f}"


def report(per_layer, meta):
    """Print a per-layer table + aggregate, return the aggregate dict."""
    print(f"\nSource: {meta['source']}   layers={len(per_layer)}   "
          f"H={meta['H']} D={meta['D']} seq={meta['seq']} "
          f"prompt={meta['prompt_len']} SRAM={meta['sram']} STT={meta['stt']}")
    hdr = (f"{'L':>3}{'Acc(all)':>10}{'Acc(anc)':>10}{'Acc(loc)':>10}"
           f"{'GOPs':>9}{'orclGOP':>9}{'prom':>6}{'demo':>6}{'drop':>6}"
           f"{'pkSRAM':>8}{'lat_us':>10}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(per_layer):
        print(f"{i:>3}{_fmt(r['acc_all']):>10}{_fmt(r['acc_anchor']):>10}"
              f"{_fmt(r['acc_local']):>10}{r['tiered_GOPs']:>9.3f}"
              f"{r['oracle_GOPs']:>9.3f}{r['promoted']:>6}{r['demoted']:>6}"
              f"{r['dropped']:>6}{r['peak_sram_tokens']:>8}"
              f"{r['latency_us']:>10.2f}")

    n = len(per_layer)
    def mean(key):
        vals = [r[key] for r in per_layer if not (isinstance(r[key], float) and r[key] != r[key])]
        return sum(vals) / len(vals) if vals else float("nan")
    def total(key):
        return sum(r[key] for r in per_layer)

    agg = {
        "num_layers": n,
        "acc_all_mean": mean("acc_all"),
        "acc_anchor_mean": mean("acc_anchor"),
        "acc_local_mean": mean("acc_local"),
        "tiered_GOPs_total": total("tiered_GOPs"),
        "oracle_GOPs_total": total("oracle_GOPs"),
        "promoted_total": total("promoted"),
        "demoted_total": total("demoted"),
        "dropped_total": total("dropped"),
        "latency_us_total": total("latency_us"),
        "energy_nj_total": total("energy_nj"),
        "peak_sram_tokens_max": max(r["peak_sram_tokens"] for r in per_layer),
        "peak_sttram_tokens_max": max(r["peak_sttram_tokens"] for r in per_layer),
    }
    saved = 1.0 - agg["tiered_GOPs_total"] / agg["oracle_GOPs_total"] \
        if agg["oracle_GOPs_total"] else float("nan")
    print("-" * len(hdr))
    print(f"AGGREGATE over {n} layers:")
    print(f"  mean Acc(all)={_fmt(agg['acc_all_mean'])}  "
          f"Acc(anchor)={_fmt(agg['acc_anchor_mean'])}  "
          f"Acc(local)={_fmt(agg['acc_local_mean'])}")
    print(f"  GOPs tiered={agg['tiered_GOPs_total']:.3f} vs "
          f"oracle={agg['oracle_GOPs_total']:.3f}  (compute saved {saved*100:.1f}%)")
    print(f"  migration: promoted={agg['promoted_total']} "
          f"demoted={agg['demoted_total']} dropped={agg['dropped_total']}")
    print(f"  peak occupancy: SRAM={agg['peak_sram_tokens_max']} tok  "
          f"STT={agg['peak_sttram_tokens_max']} tok")
    print(f"  derived latency={agg['latency_us_total']:.2f} us  "
          f"energy={agg['energy_nj_total']:.2f} nJ")
    agg["compute_saved_frac"] = saved
    return agg


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Option A: run the 3-tier KV cache on REAL (or synthetic-smoke) "
                    "per-layer attention tensors and report accuracy / GOPs / "
                    "migration / occupancy / derived latency+energy.")
    ap.add_argument("--smoke", action="store_true",
                    help="build synthetic per-layer tensors locally; NO transformers, "
                         "NO model download. Verifies the whole code path.")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                    help="HF model id for the real run (ignored under --smoke).")
    ap.add_argument("--text-file", default=None,
                    help="prompt text file for the real run; falls back to a builtin string.")
    ap.add_argument("--prompt-len", type=int, default=256,
                    help="prompt tokens; the captured tail past this is the decode replay.")
    ap.add_argument("--sram", type=int, default=128, help="SRAM token capacity.")
    ap.add_argument("--stt", type=int, default=256, help="STT-RAM token capacity.")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="subset of layer indices to evaluate (default: all captured).")
    ap.add_argument("--smoke-layers", type=int, default=4,
                    help="how many synthetic layers to build under --smoke.")
    ap.add_argument("--smoke-seq", type=int, default=320,
                    help="synthetic sequence length under --smoke (prompt+decode).")
    ap.add_argument("--heads", type=int, default=8, help="H for --smoke (real run uses captured H).")
    ap.add_argument("--dim", type=int, default=64, help="D for --smoke (real run uses captured D).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None, help="write full results (per-layer + aggregate) here.")
    args = ap.parse_args()

    if args.smoke:
        layers, masks = synthetic_layers(
            args.smoke_layers, args.heads, args.dim, args.smoke_seq)
        source = f"SYNTHETIC-SMOKE ({args.smoke_layers} layers, seq={args.smoke_seq})"
    else:
        if args.text_file:
            with open(args.text_file, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = ("The quick brown fox jumps over the lazy dog. " * 200)
        layers, _ = real_layers(args.model, text, args.prompt_len, args.device)
        masks = None
        source = f"REAL {args.model}"

    sel = args.layers if args.layers is not None else list(range(len(layers)))
    sel = [i for i in sel if 0 <= i < len(layers)]

    per_layer = []
    for i in sel:
        flags = masks[i] if masks is not None else None
        r, _m = eval_layer(layers[i], args.prompt_len, args.sram, args.stt, flags)
        per_layer.append(r)

    seq0 = layers[sel[0]][1].shape[1]
    meta = {"source": source, "H": layers[sel[0]][1].shape[0],
            "D": layers[sel[0]][1].shape[2], "seq": seq0,
            "prompt_len": min(args.prompt_len, seq0 - 1),
            "sram": args.sram, "stt": args.stt}
    agg = report(per_layer, meta)

    print("\n[transfer reminder] Compute/memory trade-off SHAPE transfers "
          "synthetic->real; accuracy-at-budget and migration VOLUME do NOT. "
          "Trust these numbers only from the REAL run (drop --smoke).")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "per_layer": per_layer, "aggregate": agg},
                      f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()

