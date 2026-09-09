#!/usr/bin/env python3
"""plot_ablation.py — paper-style line chart for a 1-D TieredKV ablation
sweep (STT budget, window size, etc). Reads a set of longbench_eval.py
JSON outputs and plots avg F1 vs. the swept parameter.

Usage:
  python experiments/plot_ablation.py --glob "results/sweep_stt/stt*.json" \
      --param-name "STT-RAM budget (tokens)" --extract-param stt_budget \
      --out figures/fig_ablation_stt_budget.png
"""
import argparse, glob, json, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--param-name", required=True, help="x-axis label")
    ap.add_argument("--param-regex", default=r"(\d+)", help="regex to pull the swept value from each filename")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    points = []
    for path in sorted(glob.glob(args.glob)):
        m = re.search(args.param_regex, os.path.basename(path))
        if not m:
            continue
        x = int(m.group(1))
        d = json.load(open(path))
        results = d.get("results", d)
        scores = [v["tieredkv"]["score"] for v in results.values() if "tieredkv" in v]
        if scores:
            points.append((x, sum(scores) / len(scores)))

    points.sort()
    if not points:
        print("No matching results found for", args.glob)
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, "o-", color="#F44336", linewidth=2, markersize=7)
    ax.set_xlabel(args.param_name, fontsize=12)
    ax.set_ylabel("Avg LongBench F1", fontsize=12)
    ax.set_title(f"TieredKV: accuracy vs. {args.param_name}", fontsize=13)
    ax.grid(alpha=0.3)
    for x, y in points:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")
    print("points:", points)


if __name__ == "__main__":
    main()
