#!/usr/bin/env python3
# plot_sota_comparison.py — publication-quality figures from longbench_eval.py output
#
# Generates:
#   1. Grouped bar chart: per-task F1 scores for each method
#   2. Summary table: LaTeX-ready comparison table
#   3. Radar chart: multi-dimensional method comparison

import os, sys, json, argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color palette matching paper conventions
COLORS = {
    "full": "#4CAF50",        # green (oracle)
    "streamingllm": "#2196F3", # blue
    "h2o": "#FF9800",          # orange
    "snapkv": "#9C27B0",       # purple
    "tieredkv": "#F44336",     # red (ours)
}

LABELS = {
    "full": "Full Cache",
    "streamingllm": "StreamingLLM",
    "h2o": "H₂O",
    "snapkv": "SnapKV",
    "tieredkv": "TieredKV (Ours)",
}

TASK_LABELS = {
    "qasper": "Qasper",
    "hotpotqa": "HotpotQA",
    "multifieldqa_en": "MultiFieldQA",
    "narrativeqa": "NarrativeQA",
    "2wikimqa": "2WikiMQA",
    "musique": "MuSiQue",
    "gov_report": "GovReport",
    "qmsum": "QMSum",
    "multi_news": "MultiNews",
    "trec": "TREC",
    "triviaqa": "TriviaQA",
    "samsum": "SAMSum",
}


def plot_bar_chart(data, tasks, methods, outdir):
    """Grouped bar chart: per-task F1 for each method."""
    fig, ax = plt.subplots(figsize=(12, 6))

    n_tasks = len(tasks)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_tasks)

    for i, method in enumerate(methods):
        scores = []
        for task in tasks:
            s = data.get(task, {}).get(method, {}).get("score", 0)
            scores.append(s)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, scores, bar_width,
            label=LABELS.get(method, method),
            color=COLORS.get(method, "#888"),
            edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        # Value labels on bars
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{score:.1f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
    ax.set_ylabel("Score (F1 / ROUGE-L × 100)", fontsize=12)
    ax.set_title("LongBench: TieredKV vs SOTA Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(100, ax.get_ylim()[1] + 5))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(outdir, "fig_longbench_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_radar(data, tasks, methods, outdir):
    """Radar/spider chart comparing methods across tasks."""
    n_tasks = len(tasks)
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for method in methods:
        scores = []
        for task in tasks:
            s = data.get(task, {}).get(method, {}).get("score", 0)
            scores.append(s)
        scores += scores[:1]
        ax.plot(angles, scores, "o-", linewidth=2,
                label=LABELS.get(method, method),
                color=COLORS.get(method, "#888"), markersize=4)
        ax.fill(angles, scores, alpha=0.1, color=COLORS.get(method, "#888"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_title("Multi-Task Performance Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    path = os.path.join(outdir, "fig_radar_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def print_latex_table(data, tasks, methods):
    """Print LaTeX-ready comparison table."""
    print("\n% LaTeX table (copy into paper)")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{LongBench results (F1 $\\times$ 100). Budget matched across eviction methods.}")
    cols = "l" + "c" * len(methods)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")

    header = "Task"
    for m in methods:
        name = LABELS.get(m, m).replace("₂", "$_2$")
        if m == "tieredkv":
            header += f" & \\textbf{{{name}}}"
        else:
            header += f" & {name}"
    print(header + " \\\\")
    print("\\midrule")

    for task in tasks:
        row = TASK_LABELS.get(task, task)
        scores = []
        for m in methods:
            s = data.get(task, {}).get(m, {}).get("score", 0)
            scores.append(s)
        best = max(scores)
        for s in scores:
            if abs(s - best) < 0.01 and s > 0:
                row += f" & \\textbf{{{s:.1f}}}"
            else:
                row += f" & {s:.1f}"
        print(row + " \\\\")

    # Average row
    print("\\midrule")
    avg_row = "\\textbf{Average}"
    avgs = []
    for m in methods:
        avg = np.mean([data.get(t, {}).get(m, {}).get("score", 0) for t in tasks])
        avgs.append(avg)
    best_avg = max(avgs)
    for avg in avgs:
        if abs(avg - best_avg) < 0.01:
            avg_row += f" & \\textbf{{{avg:.1f}}}"
        else:
            avg_row += f" & {avg:.1f}"
    print(avg_row + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_ascii_table(data, tasks, methods):
    """Print ASCII table to terminal."""
    header = f"{'Task':<20}"
    for m in methods:
        header += f" | {LABELS.get(m, m):>14}"
    print(header)
    print("-" * len(header))

    for task in tasks:
        row = f"{TASK_LABELS.get(task, task):<20}"
        for m in methods:
            s = data.get(task, {}).get(m, {}).get("score", 0)
            row += f" | {s:>14.2f}"
        print(row)

    # Average
    print("-" * len(header))
    avg_row = f"{'Average':<20}"
    for m in methods:
        avg = np.mean([data.get(t, {}).get(m, {}).get("score", 0) for t in tasks])
        avg_row += f" | {avg:>14.2f}"
    print(avg_row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to longbench_comparison.json")
    parser.add_argument("--outdir", default="figures/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.json) as f:
        raw = json.load(f)

    data = raw["results"]
    tasks = raw["config"]["tasks"]
    methods = raw["config"]["methods"]

    print(f"Tasks: {tasks}")
    print(f"Methods: {methods}")
    print(f"Budget: {raw['config']['budget']} tokens\n")

    print_ascii_table(data, tasks, methods)
    plot_bar_chart(data, tasks, methods, args.outdir)
    plot_radar(data, tasks, methods, args.outdir)
    print_latex_table(data, tasks, methods)


if __name__ == "__main__":
    main()
