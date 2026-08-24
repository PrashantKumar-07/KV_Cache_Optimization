"""
sweep_window.py
---------------
Sweeps sliding window sizes across multiple values and records accuracy,
compute savings, and write savings. Companion to sweep_budgets.py.

Run:
    python experiments/sweep_window.py \
        --model /data/.../mistral-7b \
        --device cuda --sram 128 --stt 256 \
        --windows 4 8 16 32 64 \
        --json results/window_sweep.json
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from tiered_kv_cache import TieredKVCache, TieredConfig
from baselines import FullAttention


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def eval_one_config(layers, prompt_len, sram, stt, window_size):
    """Run TieredKV with given window size across all layers, return aggregate."""
    sims_all, demoted_total, paid_total, saved_total = [], 0, 0, 0

    for (q_all, k_all, v_all) in layers:
        H, seq, D = k_all.shape
        split = max(1, min(prompt_len, seq - 1))
        k_p = k_all[:, :split, :]
        v_p = v_all[:, :split, :]
        q_p = q_all[:, :split, :]

        oracle = FullAttention()
        oracle.reset_prompt(k_p, v_p, q_p)

        cfg = TieredConfig(
            num_heads=H, head_dim=D,
            sink_size=4, window_size=window_size,
            sram_capacity=sram, sttram_capacity=stt,
            page_size=16, promote_top_pages=2,
            store_dram=False, inclusive=True
        )
        tiered = TieredKVCache(cfg)
        tiered.initial_bifurcation(k_p, v_p, q_p)

        pos = split
        for t in range(split, seq):
            q = q_all[:, t:t+1, :]
            nk = k_all[:, t:t+1, :]
            nv = v_all[:, t:t+1, :]
            o_out = oracle.step(q, nk, nv, pos)
            t_out = tiered.step(q, nk, nv, pos)
            sims_all.append(cos(t_out, o_out))
            pos += 1

        m = tiered.metrics
        demoted_total += m.total_demoted
        paid_total += m.total_paid_writes
        saved_total += m.total_writes_saved

    # compute GOPs
    oracle2 = FullAttention()
    tiered_gops, oracle_gops = 0.0, 0.0
    for (q_all, k_all, v_all) in layers:
        H, seq, D = k_all.shape
        split = max(1, min(prompt_len, seq - 1))
        k_p = v_p = q_p = None
        k_p = k_all[:, :split, :]
        v_p = v_all[:, :split, :]
        q_p = q_all[:, :split, :]
        oracle2.reset_prompt(k_p, v_p, q_p)
        cfg = TieredConfig(num_heads=H, head_dim=D, sink_size=4, window_size=window_size,
                           sram_capacity=sram, sttram_capacity=stt,
                           page_size=16, promote_top_pages=2, store_dram=False, inclusive=True)
        tiered2 = TieredKVCache(cfg)
        tiered2.initial_bifurcation(k_p, v_p, q_p)
        pos = split
        for t in range(split, seq):
            q = q_all[:, t:t+1, :]
            nk = k_all[:, t:t+1, :]
            nv = v_all[:, t:t+1, :]
            oracle2.step(q, nk, nv, pos)
            tiered2.step(q, nk, nv, pos)
            pos += 1
        tiered_gops += tiered2.metrics.total_gops
        oracle_gops += oracle2.total_macs * 2 / 1e9

    acc = sum(sims_all) / len(sims_all) if sims_all else 0.0
    compute_saved = 1.0 - (tiered_gops / oracle_gops) if oracle_gops > 0 else 0.0
    write_saved_pct = (saved_total / demoted_total * 100) if demoted_total > 0 else 100.0

    return {
        "accuracy": acc,
        "compute_saved_frac": compute_saved,
        "write_savings_pct": write_saved_pct,
        "tiered_gops": tiered_gops,
        "oracle_gops": oracle_gops,
        "demoted": demoted_total,
        "paid_writes": paid_total,
        "writes_saved": saved_total,
    }


def load_real_layers(model_path, prompt_len, device):
    """Load Mistral/Llama and capture real KV tensors."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    orig_sdpa = F.scaled_dot_product_attention
    calls = []

    def patched(q, k, v, *args, **kwargs):
        qc = q.detach().float().cpu().clone()
        kc = k.detach().float().cpu().clone()
        vc = v.detach().float().cpu().clone()
        if qc.size(1) != kc.size(1):
            n_rep = qc.size(1) // kc.size(1)
            kc = kc.repeat_interleave(n_rep, dim=1)
            vc = vc.repeat_interleave(n_rep, dim=1)
        calls.append((qc, kc, vc))
        return orig_sdpa(q, k, v, *args, **kwargs)

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()

    text = "The quick brown fox jumps over the lazy dog. " * 300
    ids = tok(text, return_tensors="pt", truncation=True, max_length=prompt_len).input_ids.to(device)

    F.scaled_dot_product_attention = patched
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    F.scaled_dot_product_attention = orig_sdpa

    layers = [(q[0], k[0], v[0]) for (q, k, v) in calls]
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model path for real run")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--sram", type=int, default=128, help="Tier 1 VRAM budget (tokens)")
    ap.add_argument("--stt", type=int, default=256, help="Tier 2 STT-RAM budget (tokens)")
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128],
                    help="Sliding window sizes to sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="results/window_sweep.json")
    args = ap.parse_args()

    print(f"Loading real model from {args.model}...")
    layers = load_real_layers(args.model, args.prompt_len, args.device)
    source = f"REAL {args.model}"

    print(f"\nSource: {source}")
    print(f"SRAM={args.sram} STT={args.stt}  Sweeping windows: {args.windows}\n")

    results = []
    print(f"{'Window':<10} | {'Accuracy':<10}")
    print("-" * 25)

    for w in args.windows:
        r = eval_one_config(layers, args.prompt_len, args.sram, args.stt, w)
        r["window_size"] = w
        results.append(r)
        print(f"{w:<10} | {r['accuracy']:<10.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    out = {"source": source, "sram": args.sram, "stt": args.stt,
           "prompt_len": args.prompt_len, "results": results}
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.json}")


if __name__ == "__main__":
    main()
