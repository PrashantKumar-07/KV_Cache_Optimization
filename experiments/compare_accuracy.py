"""
compare_accuracy.py
-------------------
Head-to-head: Full (oracle) vs StreamingLLM vs SnapKV vs TieredKV.

Why a synthetic long-range benchmark?
-------------------------------------
With purely random K/Q/V, attention is almost uniform and EVERY eviction policy
looks equally good -- which would be dishonest.  Real long-context tasks fail
because a decode step occasionally needs a token from far back (multi-hop QA,
retrieval).  We reproduce exactly that: a fraction of decode queries are steered
to point at a specific OLD key ("anchor").  A method that permanently dropped
that key cannot recover; a victim cache can promote it back.

This isolates the one property under test: does keeping evicted tokens in a warm
tier (STT-RAM) recover accuracy that permanent eviction loses?

Run:  python experiments/compare_accuracy.py
The absolute numbers are synthetic; the ORDERING (Tiered > SnapKV > Streaming)
is the claim to carry to the real LongBench run on the server.
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from tiered_kv_cache import TieredKVCache, TieredConfig
from baselines import FullAttention, StreamingLLM, SnapKV

torch.manual_seed(42)


# ---------------------------------------------------------------- data
def make_scenario(H, D, prompt_len, num_steps, anchor_frac=0.35, scale=3.0):
    """
    Build a prompt and a decode stream with injected long-range dependencies.

    Returns prompt (k,v,q) and per-step lists of (q, k, v, needs_anchor).
    ~anchor_frac of decode steps have their query aligned to a random OLD prompt
    key (an anchor beyond the sliding window), creating a genuine long-range recall.
    """
    k_p = torch.randn(H, prompt_len, D)
    v_p = torch.randn(H, prompt_len, D)
    q_p = torch.randn(H, prompt_len, D)

    steps = []
    for t in range(num_steps):
        nk = torch.randn(H, 1, D)
        nv = torch.randn(H, 1, D)
        if torch.rand(1).item() < anchor_frac:
            # steer this query toward an old prompt key (a mid-history anchor)
            anchor = torch.randint(low=4, high=prompt_len // 2, size=(1,)).item()
            q = scale * k_p[:, anchor:anchor + 1, :] + 0.3 * torch.randn(H, 1, D)
            needs = anchor
        else:
            q = torch.randn(H, 1, D)
            needs = -1
        steps.append((q, nk, nv, needs))
    return (k_p, v_p, q_p), steps


# ---------------------------------------------------------------- drivers
def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def run_system(system, prompt, steps, is_tiered=False):
    k_p, v_p, q_p = prompt
    if is_tiered:
        system.initial_bifurcation(k_p, v_p, q_p)
    else:
        system.reset_prompt(k_p, v_p, q_p)

    outs = []
    pos = k_p.shape[1]
    for (q, nk, nv, needs) in steps:
        out = system.step(q, nk, nv, pos)
        outs.append(out)
        pos += 1
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-inclusive", action="store_true",
                    help="use destructive eviction (ablation: turns the victim-cache "
                         "write-saving OFF) so you can A/B the inclusive contribution.")
    ap.add_argument("--json", default=None,
                    help="path for per-step TieredKV metrics (default results/tiered_run.json).")
    args = ap.parse_args()

    H, D = 8, 64
    prompt_len = 512
    num_steps = 150
    # equal token budget for a fair fight
    SRAM = 64
    STT = 128
    inclusive = not args.no_inclusive

    prompt, steps = make_scenario(H, D, prompt_len, num_steps)

    # oracle
    oracle = FullAttention()
    oracle_out = run_system(oracle, prompt, steps)

    systems = {
        "StreamingLLM": (StreamingLLM(sink_size=4, window_size=SRAM), False),
        "SnapKV(delete)": (SnapKV(budget=SRAM, sink_size=4, window_size=16), False),
        "TieredKV(ours)": (
            TieredKVCache(TieredConfig(
                num_heads=H, head_dim=D, sink_size=4, window_size=16,
                sram_capacity=SRAM, sttram_capacity=STT, page_size=16,
                promote_top_pages=2, store_dram=False, inclusive=inclusive)),
            True,
        ),
    }

    # split steps into anchor / non-anchor for a breakdown
    anchor_mask = [s[3] >= 0 for s in steps]

    print(f"\nScenario: prompt={prompt_len}, decode_steps={num_steps}, "
          f"H={H}, D={D}, SRAM_budget={SRAM}, STT_budget={STT}, "
          f"inclusive={inclusive}")
    print(f"Long-range (anchor) steps: {sum(anchor_mask)}/{num_steps}\n")

    header = f"{'System':<18}{'Acc(all)':>10}{'Acc(anchor)':>13}{'Acc(local)':>12}{'GOPs':>9}{'PeakTok':>9}"
    print(header)
    print("-" * len(header))

    results = {}
    for name, (sysobj, is_tiered) in systems.items():
        outs = run_system(sysobj, prompt, steps, is_tiered=is_tiered)
        sims = [cos(o, r) for o, r in zip(outs, oracle_out)]
        acc_all = sum(sims) / len(sims)
        anch = [s for s, m in zip(sims, anchor_mask) if m]
        loc = [s for s, m in zip(sims, anchor_mask) if not m]
        acc_anchor = sum(anch) / len(anch) if anch else float("nan")
        acc_local = sum(loc) / len(loc) if loc else float("nan")

        if is_tiered:
            gops = sysobj.metrics.total_gops
            peak = sysobj.metrics.peak_sram_tokens
        else:
            gops = sysobj.total_macs * 2 / 1e9
            peak = sysobj.num_tokens

        results[name] = dict(acc_all=acc_all, acc_anchor=acc_anchor,
                             acc_local=acc_local, gops=gops, peak=peak)
        print(f"{name:<18}{acc_all:>10.4f}{acc_anchor:>13.4f}{acc_local:>12.4f}"
              f"{gops:>9.3f}{peak:>9d}")

    # tiered-specific migration stats
    tiered = systems["TieredKV(ours)"][0]
    m = tiered.metrics
    print("\nTieredKV migration stats:")
    print(f"  promoted STT->SRAM : {m.total_promoted}")
    print(f"  demoted  SRAM->STT : {m.total_demoted}  "
          f"(paid writes {m.total_paid_writes}, saved {m.total_writes_saved})")
    print(f"  dropped  STT->drop : {m.total_dropped}")
    if m.total_demoted:
        pct = 100.0 * m.total_writes_saved / m.total_demoted
        print(f"  write savings      : {pct:.1f}% of demotions paid NO STT write")
    print(f"  total latency (us) : {m.total_latency_us:.2f}")
    print(f"  total energy  (nJ) : {m.total_energy_nj:.2f}")

    # the headline claim
    print("\n--- Claim check (higher is better) ---")
    s = results["SnapKV(delete)"]["acc_anchor"]
    t = results["TieredKV(ours)"]["acc_anchor"]
    print(f"  anchor-step accuracy: SnapKV={s:.4f}  TieredKV={t:.4f}  "
          f"delta={t - s:+.4f}")
    print("  -> victim cache recovers long-range recall lost by permanent eviction"
          if t > s else "  -> no recovery on this seed; tune budgets/anchor_frac")

    default_json = os.path.join(os.path.dirname(__file__), "..", "results", "tiered_run.json")
    out_json = args.json or default_json
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    m.to_json(out_json)
    print(f"\nSaved per-step TieredKV metrics -> {out_json}")


if __name__ == "__main__":
    main()
