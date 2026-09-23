#!/usr/bin/env python3
"""plot_frac_compare.py — paired frac=1.0 vs frac=0.75 figure (TieredKV only).

Panel A: per-task accuracy at both bifurcation fractions (the point: equal).
Panel B: per-task write-savings % at both fractions (the improvement).

All numbers come from committed/verified JSON result files:
  frac 1.0 : results/e2_frac1_0_short.json (5 tasks x 25 samples)
             results/final_equal_vram.json (gov_report x 10; identical config)
  frac 0.75: results/e2_frac0_75_short.json (5 tasks x 25 samples)
             results/frac075_gov.json (gov_report x 10, fresh GPU run)

Common config: VRAM=1024, STT=2048, bf16. Nothing hardcoded.
Output: figures_final/fig15_frac_compare.pdf/.png (new files only).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#bbbbbb",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "grid.color": "#999999",
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TASKS = ["multifieldqa_en", "hotpotqa", "triviaqa", "qasper",
         "narrativeqa", "gov_report"]
TASK_LABELS = {"multifieldqa_en": "MultiFieldQA", "hotpotqa": "HotpotQA",
               "triviaqa": "TriviaQA", "qasper": "Qasper",
               "narrativeqa": "NarrativeQA", "gov_report": "GovReport"}
# Same family as fig1/fig2: slate = status quo, brick red = improvement.
C_FRAC1 = "#5D6D7E"
C_FRAC075 = "#B03A2E"


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    f1 = load("results/e2_frac1_0_short.json")     # 5 short tasks, frac 1.0
    f75 = load("results/e2_frac0_75_short.json")   # 5 short tasks, frac 0.75
    ev = load("results/final_equal_vram.json")     # gov_report, frac 1.0
    g75 = load("results/frac075_gov.json")         # gov_report, frac 0.75

    def score(d, t):
        return d["results"][t]["tieredkv"]["score"]

    def sav_pct(d, t):
        cs = d["results"][t]["tieredkv"].get("cache_stats", {})
        pw, ws = cs.get("paid_writes", 0), cs.get("writes_saved", 0)
        return 100.0 * ws / (pw + ws) if (pw + ws) else 0.0

    acc1, acc75, sv1, sv75 = [], [], [], []
    for t in TASKS:
        d1 = f1 if t != "gov_report" else ev
        d75 = f75 if t != "gov_report" else g75
        acc1.append(score(d1, t))
        acc75.append(score(d75, t))
        sv1.append(sav_pct(d1, t))
        sv75.append(sav_pct(d75, t))

    x = np.arange(len(TASKS))
    bw = 0.34

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A — accuracy: the two settings agree.
    ax1.bar(x - bw / 2, acc1, bw, label="frac = 1.0 (finals)", color=C_FRAC1,
            edgecolor="white", linewidth=0.7, zorder=3)
    ax1.bar(x + bw / 2, acc75, bw, label="frac = 0.75", color=C_FRAC075,
            edgecolor="white", linewidth=0.7, zorder=3)
    for xi, a, b in zip(x, acc1, acc75):
        # Near-equal neighbours get a lifted second row so labels never touch.
        lift = 3.2 if abs(a - b) < 2.5 else 0.9
        ax1.text(xi - bw / 2, a + 0.9, f"{a:.1f}", ha="center", va="bottom",
                 fontsize=6.5, color="#333333")
        ax1.text(xi + bw / 2, b + lift, f"{b:.1f}", ha="center", va="bottom",
                 fontsize=6.5, color="#333333", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=18,
                        ha="right", fontsize=9)
    ax1.set_ylabel(r"TieredKV Score (F1 / ROUGE-L $\times$ 100)")
    ax1.set_title("(a) Accuracy holds at frac = 0.75", fontweight="bold")
    ax1.set_ylim(0, max(acc1 + acc75) * 1.22)
    ax1.legend(fontsize=8.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B — write savings: the improvement.
    ax2.bar(x - bw / 2, sv1, bw, label="frac = 1.0 (finals)", color=C_FRAC1,
            edgecolor="white", linewidth=0.7, zorder=3)
    ax2.bar(x + bw / 2, sv75, bw, label="frac = 0.75", color=C_FRAC075,
            edgecolor="white", linewidth=0.7, zorder=3)
    for xi, a, b in zip(x, sv1, sv75):
        ax2.text(xi - bw / 2, a + 1.2, f"{a:.1f}%", ha="center", va="bottom",
                 fontsize=7, color="#333333")
        ax2.text(xi + bw / 2, b + 1.2, f"{b:.1f}%", ha="center", va="bottom",
                 fontsize=7, color="#333333", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=18,
                        ha="right", fontsize=9)
    ax2.set_ylabel("STT-RAM Write Savings (%)")
    ax2.set_title("(b) Write savings at frac = 0.75", fontweight="bold")
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Bifurcation Fraction: Same Accuracy, Far Fewer Paid Writes "
        "(TieredKV, VRAM=1024 + STT=2048)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    os.makedirs("figures_final", exist_ok=True)
    fig.savefig("figures_final/fig15_frac_compare.pdf")
    fig.savefig("figures_final/fig15_frac_compare.png")
    print("  saved figures_final/fig15_frac_compare.pdf/.png")
    plt.close(fig)

    print(f"{'task':16s}{'acc 1.0':>9s}{'acc .75':>9s}{'sav 1.0':>9s}{'sav .75':>9s}")
    for t, a, b, s, w in zip(TASKS, acc1, acc75, sv1, sv75):
        print(f"{t:16s}{a:9.2f}{b:9.2f}{s:8.1f}%{w:8.1f}%")


if __name__ == "__main__":
    main()
