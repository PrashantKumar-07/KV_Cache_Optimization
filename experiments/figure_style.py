#!/usr/bin/env python3
"""figure_style.py — shared matplotlib style for the final figures (Fig. 1–9).

This module holds only the typography, palettes, labels and grouped-bar
helper that the current renderers in results/figures_20261118_renumbered/
use. The code is copied verbatim from the legacy experiments/plot_figures_v2.py
(lines 15–260, with only figure-number comments updated, of the version pinned in results/full_evaluation_20261015/plan.json
and stored in that run's source.tar.gz). The legacy figure functions, which
wrote the old figure numbers, were removed on 18 Nov 2026; the renumbering
proof shows the final figures are byte-identical to the October outputs.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cost_model import CostModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import numpy as np

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#bbbbbb",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "grid.color": "#999999",
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = "figures_final"
TAB = "results/tables"
TASKS = ["multifieldqa_en", "hotpotqa", "triviaqa", "qasper", "narrativeqa", "gov_report"]
TASK_LABELS = {"multifieldqa_en": "MultiFieldQA", "hotpotqa": "HotpotQA",
               "triviaqa": "TriviaQA", "qasper": "Qasper",
               "narrativeqa": "NarrativeQA", "gov_report": "GovReport"}
METHODS = ["full", "streamingllm", "h2o", "snapkv", "tieredkv"]
MLABEL = {"full": "Full Cache", "streamingllm": "StreamingLLM", "h2o": "H2O",
          "snapkv": "SnapKV", "tieredkv": "TieredKV (Ours)"}
# Conference-style palette: quiet slate baselines + one brick-red accent
# for TieredKV (Ours). Used by grouped_accuracy(style="pro") (Fig. 1, Fig. 2) and other renderer bar charts.
# Every other figure uses the shared PALETTE below, unchanged.
FIG1_PALETTE = {"full": "#2F3B4C", "streamingllm": "#5D6D7E", "h2o": "#85929E",
                "snapkv": "#AEB6BF", "tieredkv": "#B03A2E"}
PALETTE = {"full": "#2a7f62", "streamingllm": "#3b8bc0", "h2o": "#e07a5f",
           "snapkv": "#8e6bbf", "tieredkv": "#c0392b"}
HATCH = {"full": "", "streamingllm": "///", "h2o": "...", "snapkv": "xxx",
         "tieredkv": ""}
# Mistral-7B-Instruct-v0.2 (GQA): 32 layers x 8 KV heads x 128 dim x 2 (K,V)
LAYERS, KV_HEADS, HEAD_DIM = 32, 8, 128
BYTES_PER_LAYER_TOKEN = KV_HEADS * HEAD_DIM * 2 * 2  # bf16 = 4096 B
BYTES_PER_TOKEN = BYTES_PER_LAYER_TOKEN * LAYERS      # 131072 B = 128 KiB


def load(p):
    with open(p) as f:
        return json.load(f)


def se_of(entry):
    ss = entry.get("sample_scores", [])
    n = len(ss)
    if n < 2:
        return 0.0
    mean = sum(ss) / n
    return math.sqrt(sum((x - mean) ** 2 for x in ss) / (n - 1)) / math.sqrt(n)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig(os.path.join(OUT, stem + ".png"))
    print("  saved", stem)
    plt.close(fig)


def bar_labels(ax, xs, heights, fmt="{:.1f}", fs=7.5, base=4, step=9):
    """Value label above every bar; greedy vertical de-collision per group."""
    placed = []
    order = sorted(range(len(xs)), key=lambda i: heights[i])
    offs = {}
    for i in order:
        o = base
        while any(abs((heights[i] + o) - p) < step for p in placed):
            o += step
        offs[i] = o
        placed.append(heights[i] + o)
    span = max(heights) - min(heights) if len(heights) > 1 else 1.0
    for i, (x, h) in enumerate(zip(xs, heights)):
        ax.text(x, h + offs[i] * span / 400.0, fmt.format(h), ha="center",
                va="bottom", fontsize=fs, color="#222222")


def _decollide_text_groups(fig, ax, groups, step, max_iter=20):
    """Push overlapping value labels up until no bounding boxes overlap.

    Groups are independent (one per task cluster). Within a group, labels
    are processed left-to-right: the left label stays, the right one moves
    up by ``step`` (data coords). Uses real rendered bounding boxes with a
    small padding margin, so the result holds for any score combination.
    """
    fig.canvas.draw()
    for _ in range(max_iter):
        ylim = ax.get_ylim()
        rng = ylim[1] - ylim[0]
        ax_h_pts = ax.get_window_extent().height
        dy_pts = step / rng * ax_h_pts if rng > 0 else 6.0
        moved = False
        for g in groups:
            boxes = [t.get_window_extent().expanded(1.05, 1.08) for t in g]
            for j in range(len(g)):
                for k in range(j):
                    if boxes[j].overlaps(boxes[k]):
                        x, y = g[j].get_position()
                        g[j].set_position((x, y + step))
                        boxes[j] = boxes[j].translated(0, dy_pts)
                        moved = True
                        break
        if not moved:
            break
        fig.canvas.draw()


def grouped_accuracy(data, stem, title, note, style="legacy"):
    if style == "pro":
        _grouped_accuracy_pro(data, stem, title, note)
        return
    tasks = data["config"]["tasks"]
    n_m, n_t = len(METHODS), len(tasks)
    bw = 0.78 / n_m
    x = np.arange(n_t)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for j, m in enumerate(METHODS):
        hs, es = [], []
        for t in tasks:
            e = data["results"][t][m]
            hs.append(e["score"])
            es.append(se_of(e))
        xs = x + (j - (n_m - 1) / 2) * bw
        ax.bar(xs, hs, bw * 0.94, label=MLABEL[m], color=PALETTE[m],
               hatch=HATCH[m], edgecolor="white" if not HATCH[m] else "#333333",
               linewidth=0.6, yerr=es, error_kw={"capsize": 2, "elinewidth": 0.9,
                                                 "ecolor": "#333333"}, zorder=3)
        bar_labels(ax, xs, hs)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
    ax.set_ylabel("Score (F1 / ROUGE-L $\\times$ 100)")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylim(0, max(e["score"] for t in tasks for e in
                       [data["results"][t][m] for m in METHODS]) * 1.22)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#555555")
    despine(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, stem)


def _grouped_accuracy_pro(data, stem, title, note):
    # Matched to fig6_modeled_traffic_time geometry/style so Fig. 1/2 and Fig. 6
    # read as one family: identical palette, hatch, edge, group stride
    # (0.68 slot), bar fill (0.92 of sub-slot), figsize, legend, caption,
    # tick/title sizes and spine handling. Only the y-scale differs
    # (linear scores here vs. log time in Fig. 6).
    tasks = data["config"]["tasks"]
    n_m, n_t = len(METHODS), len(tasks)
    bw = 0.68 / n_m
    bar_w = bw * 0.92
    x = np.arange(n_t)
    FIG1_HATCH = {"full": "", "streamingllm": "", "h2o": "",
                  "snapkv": "///", "tieredkv": ""}
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for j, m in enumerate(METHODS):
        hs, es = [], []
        for t in tasks:
            e = data["results"][t][m]
            hs.append(e["score"])
            es.append(se_of(e))
        xs = x + (j - (n_m - 1) / 2) * bw
        # Identical edge for EVERY bar (hatch colour follows edgecolor,
        # so SnapKV's hatch renders white-on-pale — clearly visible and
        # exactly as wide as every other bar).
        ax.bar(xs, hs, bar_w, label=MLABEL[m], color=FIG1_PALETTE[m],
               hatch=FIG1_HATCH[m], edgecolor="white", linewidth=0.7,
               yerr=es, error_kw={"capsize": 2, "elinewidth": .7,
                                  "ecolor": "#333333"}, zorder=3)
    ymax = max(e["score"] + se_of(e) for t in tasks for e in
               [data["results"][t][m] for m in METHODS])
    ax.set_ylim(0, ymax * 1.18)
    top = ymax * 1.18
    # Professional value labels: vertical (90°), bold, ONE uniform offset
    # above every bar. Rotated text is only ~7pt wide while bars are ~12pt
    # apart, so neighbours can never overlap horizontally — no stagger or
    # staircase needed; every label sits at the same distance from its bar.
    y0, step = top * 0.015, top * 0.034
    group_texts = []
    for ti in range(n_t):
        grp = [data["results"][tasks[ti]][mm]["score"] for mm in METHODS]
        g = []
        for j, m in enumerate(METHODS):
            h = grp[j]
            err = se_of(data["results"][tasks[ti]][m])
            xp = x[ti] + (j - (n_m - 1) / 2) * bw
            t = ax.text(xp, h + err + y0, f"{h:.1f}", ha="center", va="bottom",
                        rotation=90, fontsize=7, fontweight="bold",
                        color="#222222")
            g.append(t)
        group_texts.append(g)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_ylabel("Score (F1 / ROUGE-L $\\times$ 100)", fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend below the axes (not inside): the inside upper-left box sat over
    # the leftmost group and hid pushed-up labels (e.g. MultiFieldQA
    # TieredKV). Below-axes placement can never occlude data or labels.
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, fontsize=8)
    # Caption below the axes (not inside them) so it can never overlap
    # the legend — same as Fig. 6.
    fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="#555555",
             style="italic")
    despine(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # Resolve any label-label overlaps using real rendered boxes (must run
    # after layout so box sizes/positions are final).
    _decollide_text_groups(fig, ax, group_texts, step)
    savefig(fig, stem)


