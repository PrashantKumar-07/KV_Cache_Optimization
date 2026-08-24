"""
plot_optimization.py
--------------------
Reads the VRAM sweep and window sweep JSON results and produces two
publication-quality figures:

  Figure 1: Optimal VRAM (Tier 1) size
    - 3 subplots: Accuracy vs VRAM, Compute Saved vs VRAM, Write Savings vs VRAM
    - Shaded "acceptable accuracy" region (>95%)
    - Annotated optimal point

  Figure 2: Optimal Sliding Window size
    - Same 3 subplots for window sweep

Usage:
    python experiments/plot_optimization.py \
        --vram-json results/sweep_vram_combined.json \
        --window-json results/window_sweep.json \
        --outdir figures/
"""
import os, sys, json, argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "accuracy":      "#4C72B0",
    "compute_saved": "#DD8452",
    "write_savings": "#55A868",
    "optimal":       "#C44E52",
    "threshold":     "#ABABAB",
}

ACC_THRESHOLD = 0.95   # 95% accuracy = acceptable threshold


def find_optimal_vram(xs, accs, gops):
    """
    Optimal VRAM = knee-of-curve trade-off point (reaching >97% accuracy with high compute savings).
    """
    for i, (x, acc) in enumerate(zip(xs, accs)):
        if acc >= 0.97:
            return i, x
    return int(np.argmax(accs)), xs[int(np.argmax(accs))]


def find_optimal_window(xs, accs):
    """
    Optimal window size = window size that achieves maximum accuracy.
    """
    best_i = int(np.argmax(accs))
    return best_i, xs[best_i]


def make_figure(xs, accs, compute_saved_fracs, write_savings_pcts, gops, writes,
                xlabel, title, optimal_idx, optimal_x, outpath):
    fig = plt.figure(figsize=(14, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    datasets = [
        (accs,   "Accuracy (cosine vs. Oracle)",    COLORS["accuracy"],      "Accuracy",       (min(0.9, min(accs)-0.02), 1.02)),
        (gops,   "Compute Cost (Tiered GOPs)",      COLORS["compute_saved"], "GOPs",           (0, max(gops)*1.2)),
        (writes, "STT Writes (Tokens Paid)",        COLORS["write_savings"], "Tokens Written", (0, max(max(writes)*1.2, 50))),
    ]

    for ax, (ys, subplot_title, color, ylabel, ylim) in zip(axes, datasets):
        ax.plot(xs, ys, marker="o", color=color, linewidth=2, markersize=6)
        ax.fill_between(xs, ys, alpha=0.12, color=color)
        ax.set_title(subplot_title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(ylim)
        ax.set_xticks(xs)

        # Mark threshold line on accuracy plot
        if ylabel == "Accuracy":
            ax.axhline(ACC_THRESHOLD, color=COLORS["threshold"], linestyle="--",
                       linewidth=1.2, label=f"Threshold ({ACC_THRESHOLD:.0%})")
            ax.legend(fontsize=8)

        # Highlight optimal point
        opt_y = ys[optimal_idx]
        ax.axvline(optimal_x, color=COLORS["optimal"], linestyle=":",
                   linewidth=1.5, alpha=0.7)
        ax.scatter([optimal_x], [opt_y], color=COLORS["optimal"],
                   s=90, zorder=5, label=f"Optimal: {optimal_x}")
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


def load_vram_sweep_from_per_file(base_dir, budgets):
    """Load individual per-budget JSON files produced by sweep_budgets.py."""
    xs, accs, compute_saveds, write_pcts, gops, writes = [], [], [], [], [], []
    for sram in budgets:
        path = os.path.join(base_dir, f"sweep_vram_{sram}.json")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        agg = data["aggregate"]
        acc = agg["acc_all_mean"]
        
        # calculate derived values for find_optimal
        tiered_gops = agg["tiered_GOPs_total"]
        oracle_gops = agg["oracle_GOPs_total"]
        compute_saved = 1.0 - (tiered_gops / oracle_gops) if oracle_gops > 0 else 0
        demoted = agg["demoted_total"]
        writes_saved = agg["writes_saved_total"]
        write_pct = (writes_saved / demoted * 100) if demoted > 0 else 100.0
        paid = demoted - writes_saved

        xs.append(sram)
        accs.append(acc)
        compute_saveds.append(compute_saved)
        write_pcts.append(write_pct)
        gops.append(tiered_gops)
        writes.append(paid)
    return xs, accs, compute_saveds, write_pcts, gops, writes


def load_window_sweep(path):
    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    xs = [r["window_size"] for r in results]
    accs = [r["accuracy"] for r in results]
    compute_saveds = [r["compute_saved_frac"] for r in results]
    write_pcts = [r["write_savings_pct"] for r in results]
    gops = [r["tiered_gops"] for r in results]
    writes = [r["paid_writes"] for r in results]
    return xs, accs, compute_saveds, write_pcts, gops, writes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results",
                    help="Directory with sweep_vram_*.json files")
    ap.add_argument("--vram-budgets", type=int, nargs="+",
                    default=[32, 64, 128, 192, 256, 384])
    ap.add_argument("--window-json", default="results/window_sweep.json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    # ── Figure 1: VRAM sweep ────────────────────────────────────────────────
    print("Loading VRAM sweep results...")
    xs, accs, compute_saveds, write_pcts, gops, writes = load_vram_sweep_from_per_file(
        args.results_dir, args.vram_budgets)

    if xs:
        opt_i, opt_x = find_optimal_vram(xs, accs, gops)
        print(f"  Optimal VRAM budget: {opt_x} tokens (accuracy={accs[opt_i]:.4f}, "
              f"gops={gops[opt_i]:.3f}, paid_writes={writes[opt_i]} tokens)")

        make_figure(
            xs, accs, compute_saveds, write_pcts, gops, writes,
            xlabel="Tier 1 VRAM Budget (tokens)",
            title="Optimal Tier 1 (GPU VRAM) Budget  —  TieredKV on Mistral-7B-v0.1",
            optimal_idx=opt_i, optimal_x=opt_x,
            outpath=os.path.join(args.outdir, "fig_optimal_vram.png")
        )
    else:
        print("  No VRAM sweep data found, skipping Figure 1.")

    # ── Figure 2: Window sweep ──────────────────────────────────────────────
    if os.path.exists(args.window_json):
        print("Loading window sweep results...")
        xs_w, accs_w, compute_w, write_w, gops_w, writes_w = load_window_sweep(args.window_json)

        opt_i_w, opt_x_w = find_optimal_window(xs_w, accs_w)
        print(f"  Optimal window size: {opt_x_w} tokens (accuracy={accs_w[opt_i_w]:.4f}, "
              f"gops={gops_w[opt_i_w]:.3f}, paid_writes={writes_w[opt_i_w]} tokens)")

        make_figure(
            xs_w, accs_w, compute_w, write_w, gops_w, writes_w,
            xlabel="Sliding Window Size (tokens)",
            title="Optimal Sliding Window Size  —  TieredKV on Mistral-7B-v0.1",
            optimal_idx=opt_i_w, optimal_x=opt_x_w,
            outpath=os.path.join(args.outdir, "fig_optimal_window.png")
        )
    else:
        print(f"  Window sweep file not found at {args.window_json}. Run sweep_window.py first.")


if __name__ == "__main__":
    main()
