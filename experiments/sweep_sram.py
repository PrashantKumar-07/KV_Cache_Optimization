"""
sweep_sram.py  (Option B)
-------------------------
Sweep the SRAM budget and locate the "crossover point": the smallest SRAM size
at which TieredKV's compute+migration cost drops below full attention while
accuracy stays acceptable.

WHAT TRANSFERS FROM THIS SWEEP TO A REAL MODEL (read before trusting a number):
  * The SHAPE of the compute/memory trade-off (cost vs SRAM size) is a function
    of tensor sizes and the cost model -- it transfers.
  * The ACCURACY at each budget and the MIGRATION VOLUME depend on the real
    attention distribution -- they DO NOT transfer. Re-run this SAME script on
    the server with real per-layer K/V to get the paper-ready operating point.

The script is data-source agnostic: swap `make_scenario` for real captured
tensors and every metric below is computed identically.

Latency baseline (conservative): full attention is charged only for reading its
resident tokens from the *fast* SRAM tier each step -- the best possible case
for the baseline, which makes our win harder to show, not easier.

Usage:
  python experiments/sweep_sram.py
  python experiments/sweep_sram.py --budgets 16 32 64 128 256 --prompt 1024 --steps 200 --device cpu
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from tiered_kv_cache import TieredKVCache, TieredConfig
from baselines import FullAttention
from cost_model import CostModel


def make_scenario(H, D, prompt_len, num_steps, anchor_frac=0.35, scale=3.0, device="cpu", seed=42):
    """Synthetic prompt + decode stream with injected long-range dependencies.
    ~anchor_frac of steps steer the query at an OLD prompt key (a recall test)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_p = torch.randn(H, prompt_len, D, generator=g).to(device)
    v_p = torch.randn(H, prompt_len, D, generator=g).to(device)
    q_p = torch.randn(H, prompt_len, D, generator=g).to(device)
    steps = []
    for _ in range(num_steps):
        nk = torch.randn(H, 1, D, generator=g).to(device)
        nv = torch.randn(H, 1, D, generator=g).to(device)
        if torch.rand(1, generator=g).item() < anchor_frac:
            anchor = torch.randint(4, prompt_len // 2, (1,), generator=g).item()
            q = scale * k_p[:, anchor:anchor + 1, :] + 0.3 * torch.randn(H, 1, D, generator=g).to(device)
            needs = anchor
        else:
            q = torch.randn(H, 1, D, generator=g).to(device)
            needs = -1
        steps.append((q, nk, nv, needs))
    return (k_p, v_p, q_p), steps


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def run_full(prompt, steps):
    sysobj = FullAttention()
    sysobj.reset_prompt(*prompt)
    outs, pos = [], prompt[0].shape[1]
    for (q, nk, nv, _) in steps:
        outs.append(sysobj.step(q, nk, nv, pos)); pos += 1
    return outs, sysobj


def run_tiered(cfg, prompt, steps):
    cache = TieredKVCache(cfg)
    cache.initial_bifurcation(*prompt)
    outs, pos = [], prompt[0].shape[1]
    for (q, nk, nv, _) in steps:
        outs.append(cache.step(q, nk, nv, pos)); pos += 1
    return outs, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 96, 128, 192, 256])
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--stt-mult", type=float, default=2.0, help="STT capacity = mult * SRAM")
    ap.add_argument("--acc-thresh", type=float, default=0.30, help="min acceptable Acc(all)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    H, D = args.heads, args.dim
    cm = CostModel(bytes_per_elem=2)
    prompt, steps = make_scenario(H, D, args.prompt, args.steps, device=args.device)
    anchor_mask = [s[3] >= 0 for s in steps]

    # --- baseline: full attention (accuracy oracle + conservative latency ref) ---
    full_outs, full_sys = run_full(prompt, steps)
    # conservative baseline latency: read all resident tokens from SRAM each step
    full_lat_us = 0.0
    resident = args.prompt
    for _ in steps:
        resident += 1
        lat, _ = cm.sram_compute_read_cost(resident * 2 * H * D)
        full_lat_us += lat

    print(f"\nSweep: prompt={args.prompt} steps={args.steps} H={H} D={D} device={args.device}")
    print(f"Anchor(long-range) steps: {sum(anchor_mask)}/{args.steps}")
    print(f"Full-attention baseline latency (conservative): {full_lat_us:.1f} us\n")

    hdr = (f"{'SRAM':>6}{'STT':>6}{'Acc_all':>9}{'Acc_anch':>10}{'GOPs':>8}"
           f"{'Migrate':>9}{'OurLat_us':>11}{'NetSpeedup':>12}{'Verdict':>9}")
    print(hdr); print("-" * len(hdr))

    rows = []
    crossover = None
    for sram in args.budgets:
        stt = int(args.stt_mult * sram)
        cfg = TieredConfig(num_heads=H, head_dim=D, sink_size=4, window_size=16,
                           sram_capacity=sram, sttram_capacity=stt, page_size=16,
                           promote_top_pages=2, store_dram=False, device=args.device)
        outs, cache = run_tiered(cfg, prompt, steps)

        sims = [cos(o, r) for o, r in zip(outs, full_outs)]
        acc_all = sum(sims) / len(sims)
        anch = [s for s, m in zip(sims, anchor_mask) if m]
        acc_anch = sum(anch) / len(anch) if anch else float("nan")

        m = cache.metrics
        gops = m.total_gops
        migrate = m.total_promoted + m.total_demoted + m.total_dropped
        our_lat = m.total_latency_us
        net = full_lat_us - our_lat
        faster = net > 0
        good = acc_all >= args.acc_thresh
        verdict = "WIN" if (faster and good) else ("fast" if faster else "slow")
        if crossover is None and faster and good:
            crossover = sram

        print(f"{sram:>6}{stt:>6}{acc_all:>9.4f}{acc_anch:>10.4f}{gops:>8.3f}"
              f"{migrate:>9}{our_lat:>11.1f}{net:>+12.1f}{verdict:>9}")
        rows.append((sram, stt, acc_all, acc_anch, gops, migrate, our_lat, net, verdict))

    print("\n--- Crossover ---")
    if crossover is not None:
        frac = 100.0 * crossover / args.prompt
        print(f"  Smallest SRAM budget that is faster than full attention AND keeps")
        print(f"  Acc(all) >= {args.acc_thresh}:  SRAM = {crossover} tokens "
              f"(~{frac:.1f}% of the {args.prompt}-token context).")
    else:
        print("  No budget in the swept range satisfied both constraints.")
        print("  Try: lower --acc-thresh, wider --budgets, or tune promote_top_pages.")

    print("\n  REMINDER: this crossover is from SYNTHETIC data. The COMPUTE/MEMORY")
    print("  shape transfers; the ACCURACY and MIGRATION volume do NOT. Re-run this")
    print("  exact script on the server with real K/V to get the reportable number.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sram", "stt", "acc_all", "acc_anchor", "gops",
                        "migrate", "our_lat_us", "net_speedup_us", "verdict"])
            w.writerows(rows)
        print(f"\n  Wrote {args.csv}")


if __name__ == "__main__":
    main()
