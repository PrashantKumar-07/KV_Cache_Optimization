#!/usr/bin/env python3
"""
plot_publication_figures.py
───────────────────────────
Generates all publication-quality figures and LaTeX tables for the
TieredKV paper. Every number comes directly from the JSON result files
produced by longbench_eval.py — nothing is hardcoded or hand-crafted.

Figures produced
────────────────
  fig1_accuracy_comparison.pdf/.png  — grouped bar chart (equal-VRAM setting)
  fig2_latency_comparison.pdf/.png   — modeled latency per method (fair setting)
  fig3_write_savings.pdf/.png        — STT write savings vs total demotions
  (fig4 retired — see plot_figures_v2.py fig4_decode_scaling)
  fig5_stt_ablation.pdf/.png         — RETIRED (mixed ms and % on one axis;
  Fig 5 now comes from plot_figures_v2.py: fig5_clean)
  fig6_migration_profile.pdf/.png    — stacked bar: demote / promote / peek / drop
  fig7_accuracy_fair.pdf/.png        — grouped bar chart (fair setting)

Tables produced (stdout as LaTeX + written to results/tables/)
──────────────────────────────────────────────────────────────
  table1_accuracy_equivram.tex       — per-task + avg accuracy (equal-VRAM)
  table2_accuracy_fair.tex           — per-task + avg accuracy (fair)
  table3_efficiency.tex              — latency, energy, write savings (fair)
  table4_stt_ablation.tex            — accuracy vs STT budget

Usage
─────
  python experiments/plot_publication_figures.py \
      --equal-vram results/final_equal_vram.json \
      --fair        results/final_equal_attended.json \
      --sweep-glob  "results/clean_sweep_stt/stt*.json" \
      --outdir      figures/

Requirements: matplotlib, numpy  (already in venv_kvcache)
"""

import argparse
import glob
import json
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib import rcParams


# ─────────────────────────────────────────────────────────────────────────────
# Global style — NeurIPS / ICML convention
# ─────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.30,
    "grid.linewidth":     0.5,
    "grid.color":         "#999999",
    "lines.linewidth":    1.6,
    "lines.markersize":   5,
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,   # TrueType in PDF (no Type 3)
    "ps.fonttype":        42,
})

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — muted, professional palette (not primary/neon)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "full":         "#2d6a4f",   # deep teal-green  (oracle)
    "streamingllm": "#457b9d",   # steel blue
    "h2o":          "#e07a5f",   # terracotta
    "snapkv":       "#8e6bbf",   # muted violet
    "tieredkv":     "#c0392b",   # crimson (ours)
}

HATCHES = {
    "full":         "",
    "streamingllm": "///",
    "h2o":          "...",
    "snapkv":       "xxx",
    "tieredkv":     "",
}

METHOD_LABELS = {
    "full":         "Full Cache",
    "streamingllm": "StreamingLLM",
    "h2o":          r"H$_2$O",
    "snapkv":       "SnapKV",
    "tieredkv":     r"\textbf{TieredKV (Ours)}",
}

METHOD_LABELS_PLAIN = {
    "full":         "Full Cache",
    "streamingllm": "StreamingLLM",
    "h2o":          "H$_2$O",
    "snapkv":       "SnapKV",
    "tieredkv":     "TieredKV (Ours)",
}

TASK_LABELS = {
    "multifieldqa_en": "MultiFieldQA",
    "hotpotqa":        "HotpotQA",
    "triviaqa":        "TriviaQA",
    "qasper":          "Qasper",
    "narrativeqa":     "NarrativeQA",
    "gov_report":      "GovReport",
}

TASK_LABELS_LATEX = {
    "multifieldqa_en": "MultiFieldQA",
    "hotpotqa":        "HotpotQA",
    "triviaqa":        "TriviaQA",
    "qasper":          "Qasper",
    "narrativeqa":     "NarrativeQA",
    "gov_report":      "GovReport",
}

METHOD_ORDER = ["full", "streamingllm", "h2o", "snapkv", "tieredkv"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_score(data, task, method):
    return data["results"].get(task, {}).get(method, {}).get("score", None)


def get_stat(data, task, method, key):
    return data["results"].get(task, {}).get(method, {}).get("cache_stats", {}).get(key, None)


def savefig(fig, outdir, stem):
    """Save as both PNG (for quick preview) and PDF (for paper)."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  ✓  {path}")
    plt.close(fig)


def bold_best(values, fmt="{:.1f}"):
    """Return list of strings; best value gets \\textbf{}."""
    valid = [v for v in values if v is not None]
    best = max(valid) if valid else None
    out = []
    for v in values:
        if v is None:
            out.append("--")
        elif best is not None and abs(v - best) < 0.005:
            out.append(r"\textbf{" + fmt.format(v) + r"}")
        else:
            out.append(fmt.format(v))
    return out


def bold_best_low(values, fmt="{:.1f}"):
    """Bold-mark the LOWEST value (for latency tables)."""
    valid = [v for v in values if v is not None]
    best = min(valid) if valid else None
    out = []
    for v in values:
        if v is None:
            out.append("--")
        elif best is not None and abs(v - best) < 0.005:
            out.append(r"\textbf{" + fmt.format(v) + r"}")
        else:
            out.append(fmt.format(v))
    return out


def write_tex(lines, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{stem}.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 & 7 — Grouped accuracy bar charts (equal-VRAM and fair)
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy_bar(data, outdir, stem, title_tag, caption_note):
    # Styled to match fig2_latency_comparison exactly (same family):
    # identical palette / hatch / edge / group stride (0.68 slot) /
    # bar fill (0.92 of sub-slot) / figsize / legend / caption placement /
    # tick-title sizes / spine handling. Only the y-scale differs
    # (linear scores here vs. log latency in Fig 2).
    tasks   = data["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in data["config"]["methods"]]

    n_tasks   = len(tasks)
    n_methods = len(methods)
    # Group occupies 0.68 of the unit slot (clear gap between task groups);
    # each bar fills 92% of its sub-slot (slim gap between neighbours) —
    # same as Fig 2.
    bw        = 0.68 / n_methods
    bar_w     = bw * 0.92
    x         = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(9.6, 4.2))

    all_scores = []  # per-method score lists, for the per-group label pass
    for i, m in enumerate(methods):
        scores = [get_score(data, t, m) or 0.0 for t in tasks]
        all_scores.append(scores)
        offset = (i - n_methods / 2 + 0.5) * bw
        ax.bar(
            x + offset, scores, bar_w,
            label=METHOD_LABELS_PLAIN.get(m, m),
            color=FIG2_PALETTE[m],
            hatch=FIG2_HATCH[m],
            # Identical edge for EVERY bar (hatch colour follows edgecolor,
            # so SnapKV's hatch renders white-on-pale — clearly visible and
            # exactly as wide as every other bar). Same as Fig 2.
            edgecolor=FIG2_EDGE,
            linewidth=FIG2_LW,
            zorder=3,
        )

    # Professional value labels: vertical (90°), bold, ONE uniform offset
    # above every bar. Rotated text is only ~7pt wide while bars are ~12pt
    # apart, so neighbours can never overlap horizontally — no stagger or
    # staircase needed; every label sits at the same distance from its bar.
    ylim_top = 80.0
    y0, step = ylim_top * 0.015, ylim_top * 0.034
    group_texts = []
    for ti in range(n_tasks):
        grp = [all_scores[i][ti] for i in range(n_methods)]
        g = []
        for i in range(n_methods):
            sc = grp[i]
            if sc <= 0:
                continue
            xp = x[ti] + (i - n_methods / 2 + 0.5) * bw
            t = ax.text(xp, sc + y0, f"{sc:.1f}", ha="center", va="bottom",
                        rotation=90, fontsize=7, fontweight="bold",
                        color="#222222")
            g.append(t)
        group_texts.append(g)

    # Averages as text annotation in top-right
    avg_texts = []
    for m in methods:
        scores = [get_score(data, t, m) or 0.0 for t in tasks]
        avg = np.mean(scores)
        avg_texts.append(f"{METHOD_LABELS_PLAIN[m]}: {avg:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_ylabel("Score (F1 / ROUGE-L ×100)", fontsize=10)
    ax.set_title(
        f"LongBench Accuracy — TieredKV vs. SOTA Baselines ({title_tag})",
        fontsize=10.5, fontweight="bold", pad=6,
    )
    ax.set_ylim(0, 80)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(which="minor", axis="y", length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend below the axes (not inside): the inside upper-left box sat over
    # the leftmost group and hid pushed-up labels (e.g. MultiFieldQA
    # TieredKV). Below-axes placement can never occlude data or labels.
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, fontsize=8)

    # Caption below the axes (not inside them) so it can never overlap
    # the legend — same as Fig 2.
    fig.text(
        0.5, 0.005,
        caption_note,
        ha="center", fontsize=7, color="#555555", style="italic",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # Resolve any label-label overlaps using real rendered boxes (must run
    # after layout so box sizes/positions are final).
    _decollide_text_groups(fig, ax, group_texts, step)
    savefig(fig, outdir, stem)


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


def fig1_accuracy_equivram(data_ev, outdir):
    print("\n[Fig 1] Accuracy bar chart (equal-VRAM setting)…")
    _accuracy_bar(
        data_ev, outdir,
        stem="fig1_accuracy_equivram",
        title_tag="Equal-VRAM",
        caption_note="Equal-VRAM: TieredKV uses VRAM=1024 + STT-RAM=2048; baselines at budget=1024 tokens."
    )


def fig7_accuracy_fair(data_fair, outdir):
    print("\n[Fig 7] Accuracy bar chart (fair / iso-VRAM setting)…")
    _accuracy_bar(
        data_fair, outdir,
        stem="fig7_accuracy_fair",
        title_tag="Iso-VRAM (Fair)",
        caption_note="Fair comparison: all methods use identical VRAM budget (992 tokens)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Modeled latency per method (fair setting, bar chart)
# ─────────────────────────────────────────────────────────────────────────────

# Fig 2 uses fig1's palette (slate baselines + brick-red accent) so the two
# figures read as one family. The palest shade (SnapKV) gets a single
# subtle diagonal hatch so it stays distinguishable on white.
# NOTE: all bars share the SAME white edge + linewidth so the hatched bar
# can never render visually wider than its neighbours (a dark edge on only
# one bar adds ~1px of stroke each side, i.e. ~10% apparent width). The
# white hatch on the pale fill keeps SnapKV distinguishable while every
# bar keeps identical geometry: equal width, equal centre spacing.
FIG2_PALETTE = {"full": "#2F3B4C", "streamingllm": "#5D6D7E", "h2o": "#85929E",
                "snapkv": "#AEB6BF", "tieredkv": "#B03A2E"}
FIG2_HATCH = {"full": "", "streamingllm": "", "h2o": "", "snapkv": "///",
              "tieredkv": ""}
FIG2_EDGE = "white"
FIG2_LW = 0.7

def fig2_latency(data_fair, outdir):
    print("\n[Fig 2] Latency comparison (fair setting)…")
    tasks   = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in data_fair["config"]["methods"]]

    # Aggregate: total modeled latency across all tasks (in ms)
    task_latencies = {}  # method -> per-task latency list (ms)
    for m in methods:
        lats = []
        for t in tasks:
            v = get_stat(data_fair, t, m, "modeled_latency_us")
            lats.append((v / 1000.0) if v is not None else 0.0)  # us -> ms
        task_latencies[m] = lats

    n_tasks   = len(tasks)
    n_methods = len(methods)
    # Group occupies 0.68 of the unit slot (clear gap between task groups);
    # each bar fills 92% of its sub-slot (slim gap between neighbours).
    bw        = 0.68 / n_methods
    bar_w     = bw * 0.92
    x         = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(9.6, 4.2))

    for i, m in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bw
        bars = ax.bar(
            x + offset, task_latencies[m], bar_w,
            label=METHOD_LABELS_PLAIN.get(m, m),
            color=FIG2_PALETTE[m],
            hatch=FIG2_HATCH[m],
            # Identical edge for EVERY bar (hatch colour follows edgecolor,
            # so SnapKV's hatch renders white-on-pale — clearly visible and
            # exactly as wide as every other bar). Never give the hatched
            # bar a lone dark edge: it photographs ~10% wider.
            edgecolor=FIG2_EDGE,
            linewidth=FIG2_LW,
            zorder=3,
        )
        # Value on every bar: all horizontal, just above the bar top. If two
        # neighbours are nearly equal their labels would touch, so the odd
        # bar of such a group gets a slightly higher row.
        heights = task_latencies[m]
        for ti, h in enumerate(heights):
            grp = [task_latencies[mm][ti] for mm in methods]
            tight = any(
                abs(grp[k + 1] - grp[k]) / max(grp[k], 1) < 0.10
                for k in range(len(grp) - 1)
            )
            f = 1.18 + (0.22 if (tight and i % 2 == 1) else 0.0)
            ax.text(x[ti] + offset, h * f if h > 0 else 1.0, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=6.5, color="#222222")

    ax.set_yscale("log")
    ax.set_ylim(20, 5200)
    # Plain numbers on the y-axis (100, 500, 1000 …), not 10²/10³ powers;
    # minor ticks shown without labels.
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 4000])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.legend(loc="upper left", ncol=1, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_ylabel("Modeled Decode Latency (ms)", fontsize=10)
    ax.set_title(
        "Modeled Decode Latency — TieredKV vs. SOTA (Fair / Iso-VRAM)",
        fontsize=10.5, fontweight="bold", pad=6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Caption below the axes (not inside them) so it can never overlap
    # the legend.
    fig.text(
        0.5, 0.005,
        "Latency: analytical model (GPU VRAM \u2194 STT-RAM tier bandwidth). "
        "TieredKV latency includes migration overhead.",
        ha="center", fontsize=7, color="#555555", style="italic",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, outdir, "fig2_latency_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — STT Write savings: paid_writes vs writes_saved (TieredKV only)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_write_savings(data_fair, data_ev, sweep_paths, outdir):
    print("\n[Fig 3] Write savings bar chart…")

    # PREFERENCE ORDER for write-savings data:
    #  1. fair json (full 6 tasks, some have real saves)
    #  2. equal-vram json (full 6 tasks, stt_expose=32 so mostly 0% saves)
    #  3. sweep files (3 tasks, good signal at stt512 where backup reuse fires)
    #
    # We gather from fair first; supplement with sweep for richer savings signal.

    # -- From fair run (6 tasks) --
    tasks = data_fair["config"]["tasks"]
    paid_list   = []
    saved_list  = []
    task_labels = []
    for t in tasks:
        pw = get_stat(data_fair, t, "tieredkv", "paid_writes")
        ws = get_stat(data_fair, t, "tieredkv", "writes_saved")
        if pw is not None:
            paid_list.append(pw)
            saved_list.append(ws if ws is not None else 0)
            task_labels.append(TASK_LABELS.get(t, t))

    # -- Also build a line-chart from sweep (write savings % vs STT budget) --
    # NOTE: sort NUMERICALLY by the budget in the filename. Plain sorted()
    # orders stt512 after stt3072, which made the trend line zigzag back and
    # draw a phantom second line.
    def _sweep_key(path):
        m = re.search(r"(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0
    sweep_stt    = []
    sweep_ws_pct = []
    for path in sorted(sweep_paths, key=_sweep_key):
        m = re.search(r"(\d+)", os.path.basename(path))
        if not m:
            continue
        stt = int(m.group(1))
        d   = load_json(path)
        ws_pcts = []
        for task_data in d["results"].values():
            td = task_data.get("tieredkv", {})
            cs = td.get("cache_stats", {})
            pw = cs.get("paid_writes", 0)
            ws = cs.get("writes_saved", 0)
            total = pw + ws
            if total > 0:
                ws_pcts.append(100.0 * ws / total)
        if ws_pcts:
            sweep_stt.append(stt)
            sweep_ws_pct.append(np.mean(ws_pcts))

    if not paid_list:
        print("  ⚠ No write-savings data found; skipping Fig 3.")
        return

    x = np.arange(len(task_labels))
    save_pct = [100.0 * s / (p + s) if (p + s) > 0 else 0.0
                for p, s in zip(paid_list, saved_list)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))

    # Left: stacked bar — paid vs saved (per task, fair run)
    bw = 0.55
    ax1.bar(x, paid_list,  bw, label="Paid writes",  color="#c0392b", alpha=0.88, zorder=3)
    ax1.bar(x, saved_list, bw, label="Writes saved", color="#27ae60", alpha=0.88,
            bottom=paid_list, zorder=3)
    ymax1 = max(p + s for p, s in zip(paid_list, saved_list))
    ax1.set_ylim(0, ymax1 * 1.12)
    # One short line above each bar: the saved share (0% shown explicitly so
    # zero-save tasks never look like missing data). Full paid/saved
    # absolutes live in the table under the axis — no text inside or
    # stacked on the narrow bars, so nothing can overlap.
    for xi, p, s, pct in zip(x, paid_list, saved_list, save_pct):
        tot = p + s
        ax1.text(xi, tot + 0.02 * ymax1, f"{pct:.1f}% saved",
                 ha="center", va="bottom", fontsize=8.5, color="#1e8449",
                 fontweight="bold")
    # Data table: exact paid / saved counts (k = thousand token pages).
    tbl = ax1.table(
        cellText=[
            [f"{p / 1e3:.0f}k" for p in paid_list],
            [f"{s / 1e3:.0f}k" for s in saved_list],
            [f"{v:.1f}%" for v in save_pct],
        ],
        rowLabels=["Paid", "Saved", "Saved %"],
        colLabels=task_labels,
        loc="bottom", bbox=[0.0, -0.52, 1.0, 0.34],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for _, cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        cell.PAD = 0.03
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels, rotation=25, ha="right", fontsize=8.5)
    ax1.set_ylabel("STT-RAM Write Events (token pages)", fontsize=10)
    ax1.set_title("STT-RAM Write Traffic: Paid vs. Saved (Fair)", fontsize=10.5, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=8.5)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else str(int(v)))
    )

    # Right: write savings % vs STT budget (from sweep, where backup-reuse fires)
    if sweep_stt:
        ax2.plot(sweep_stt, sweep_ws_pct, "D-", color="#27ae60", linewidth=2,
                 markersize=7, zorder=5)
        for xs, ys in zip(sweep_stt, sweep_ws_pct):
            ax2.annotate(f"{ys:.0f}%", (xs, ys),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8.5, color="#27ae60")
        ax2.set_xlabel("STT-RAM Budget (tokens)", fontsize=10)
        ax2.set_ylabel("Avg. Write Savings (%)", fontsize=10)
        ax2.set_title("Write Savings vs. STT Budget (Ablation)", fontsize=10.5, fontweight="bold")
        ax2.set_ylim(0, 105)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.text(0.02, 0.97,
                 "Higher STT budget → more backup reuse → fewer paid writes.",
                 transform=ax2.transAxes, fontsize=7, color="#555555",
                 va="top", ha="left", style="italic")
    else:
        # Fall back: per-task savings % bar
        colors = ["#c0392b" if p < 10 else "#27ae60" for p in save_pct]
        ax2.bar(x, save_pct, bw, color=colors, alpha=0.88, zorder=3, edgecolor="white")
        for xi, pct in zip(x, save_pct):
            ax2.text(xi, pct + 0.8, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(task_labels, rotation=25, ha="right", fontsize=8.5)
        ax2.set_ylabel("Write Savings (%)", fontsize=10)
        ax2.set_title("STT-RAM Write Savings Rate per Task", fontsize=10.5, fontweight="bold")
        ax2.set_ylim(0, 105)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "TieredKV STT-RAM Write Efficiency",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.tight_layout()
    # Room for the data table slung under the left panel.
    fig.subplots_adjust(bottom=0.28)
    savefig(fig, outdir, "fig3_write_savings")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Efficiency scatter: accuracy vs. latency (Pareto frontier)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_efficiency_scatter(data_fair, outdir):
    print("\n[Fig 4] Accuracy vs. latency scatter (Pareto)…")
    tasks   = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in data_fair["config"]["methods"]]

    avg_accuracy = {}
    avg_latency_ms = {}
    for m in methods:
        scores = []
        lats   = []
        for t in tasks:
            sc = get_score(data_fair, t, m)
            lt = get_stat(data_fair, t, m, "modeled_latency_us")
            if sc is not None and lt is not None:
                scores.append(sc)
                lats.append(lt / 1000.0)
        if scores:
            avg_accuracy[m]    = np.mean(scores)
            avg_latency_ms[m]  = np.mean(lats)

    fig, ax = plt.subplots(figsize=(6, 4.8))

    for m in methods:
        if m not in avg_accuracy:
            continue
        ax.scatter(
            avg_latency_ms[m], avg_accuracy[m],
            s=120, color=PALETTE[m], zorder=5,
            marker="D" if m == "tieredkv" else "o",
            edgecolors="#333333", linewidths=0.6,
        )
        # Offset labels smartly
        ha_map = {
            "full": "right",
            "streamingllm": "left",
            "h2o": "right",
            "snapkv": "left",
            "tieredkv": "left",
        }
        va_map = {
            "full": "top",
            "streamingllm": "bottom",
            "h2o": "bottom",
            "snapkv": "top",
            "tieredkv": "center",
        }
        xoff_map = {
            "full": -1.2, "streamingllm": 1.2, "h2o": -1.2,
            "snapkv": 1.2, "tieredkv": 1.5,
        }
        ax.annotate(
            METHOD_LABELS_PLAIN[m],
            xy=(avg_latency_ms[m], avg_accuracy[m]),
            xytext=(xoff_map.get(m, 2), 0.4),
            textcoords="offset points",
            fontsize=8,
            ha=ha_map.get(m, "left"),
            va=va_map.get(m, "center"),
            color=PALETTE[m],
            fontweight="bold" if m == "tieredkv" else "normal",
        )

    # Ideal corner
    ax.annotate(
        "← better\n↑ better",
        xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=7.5, color="#777777", ha="left", va="top",
    )

    ax.set_xlabel("Avg. Modeled Decode Latency (ms)", fontsize=10)
    ax.set_ylabel("Avg. LongBench Score", fontsize=10)
    ax.set_title(
        "Accuracy vs. Latency — Fair Comparison",
        fontsize=10.5, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend via proxy
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PALETTE["tieredkv"],
               markeredgecolor="#333", markersize=8, label="TieredKV (Ours)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markeredgecolor="#333", markersize=7, label="Baselines"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    savefig(fig, outdir, "fig4_efficiency_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — STT budget ablation: accuracy + latency vs STT size (dual y-axis)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_stt_ablation(sweep_paths, outdir):
    print("\n[Fig 5] STT budget ablation…")
    if not sweep_paths:
        print("  ⚠ No sweep files found; skipping Fig 5.")
        return

    points = []  # (stt_budget, avg_accuracy, avg_latency_ms, avg_write_savings_pct)
    for path in sorted(sweep_paths):
        m = re.search(r"(\d+)", os.path.basename(path))
        if not m:
            continue
        stt = int(m.group(1))
        d = load_json(path)
        results = d.get("results", {})
        scores, lats, wsavings = [], [], []
        for task_data in results.values():
            td = task_data.get("tieredkv", {})
            sc = td.get("score")
            cs = td.get("cache_stats", {})
            lt = cs.get("modeled_latency_us")
            ws = cs.get("write_savings_pct")
            if sc is not None:
                scores.append(sc)
            if lt is not None:
                lats.append(lt / 1000.0)
            if ws is not None:
                wsavings.append(ws)
        if scores:
            points.append((
                stt,
                np.mean(scores),
                np.mean(lats) if lats else 0.0,
                np.mean(wsavings) if wsavings else 0.0,
            ))

    points.sort()
    if not points:
        print("  ⚠ No valid sweep results; skipping Fig 5.")
        return

    xs       = [p[0] for p in points]
    acc      = [p[1] for p in points]
    lats     = [p[2] for p in points]
    wsavings = [p[3] for p in points]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(xs, acc,  "D-", color="#c0392b",  linewidth=2,   markersize=7,
                   label="Avg. Accuracy (F1)", zorder=5)
    ln2 = ax2.plot(xs, lats, "s--", color="#457b9d", linewidth=1.8, markersize=6,
                   label="Avg. Latency (ms)", zorder=5)
    ln3 = ax2.plot(xs, wsavings, "^:", color="#2d6a4f", linewidth=1.5, markersize=6,
                   label="Write Savings (%)", zorder=5)

    # Annotate accuracy
    for x, y in zip(xs, acc):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="#c0392b")

    ax1.set_xlabel("STT-RAM Budget (tokens, VRAM fixed at 1024)", fontsize=10)
    ax1.set_ylabel("Avg. LongBench Score", fontsize=10, color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylabel("Latency (ms) / Write Savings (%)", fontsize=10, color="#457b9d")
    ax2.tick_params(axis="y", labelcolor="#457b9d")
    ax1.set_title(
        "TieredKV: Accuracy & Efficiency vs. STT-RAM Budget",
        fontsize=10.5, fontweight="bold", pad=6,
    )
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=8, loc="lower right")

    fig.tight_layout()
    savefig(fig, outdir, "fig5_stt_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Migration profile: stacked bar per task (demote/promote/peek/drop)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_migration_profile(data_ev, outdir):
    print("\n[Fig 6] Migration profile (stacked bar)…")
    tasks = data_ev["config"]["tasks"]

    demoted   = []
    promoted  = []
    peeked    = []
    dropped   = []
    tl        = []

    for t in tasks:
        dem = get_stat(data_ev, t, "tieredkv", "demoted")
        pro = get_stat(data_ev, t, "tieredkv", "promoted")
        # "peeked" (in-place attention) never happens on the harness path --
        # those events are hysteresis-denied promotion candidates, reported
        # as promotions_deferred (older files: peeked). Label honestly.
        pee = get_stat(data_ev, t, "tieredkv", "promotions_deferred")
        if pee is None:
            pee = get_stat(data_ev, t, "tieredkv", "peeked")
        drp = get_stat(data_ev, t, "tieredkv", "dropped")
        if dem is None:
            continue
        demoted.append(dem)
        promoted.append(pro or 0)
        peeked.append(pee or 0)
        dropped.append(drp or 0)
        tl.append(TASK_LABELS.get(t, t))

    if not demoted:
        print("  ⚠ No migration stats; skipping Fig 6.")
        return

    x  = np.arange(len(tl))
    bw = 0.55

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    p1 = ax.bar(x, demoted,  bw, label="Demoted (VRAM→STT)",  color="#e07a5f", zorder=3)
    p2 = ax.bar(x, promoted, bw, label="Promoted (STT→VRAM)", color="#457b9d", zorder=3,
                bottom=demoted)
    bot2 = [a + b for a, b in zip(demoted, promoted)]
    p3 = ax.bar(x, peeked,   bw, label="Deferred promotions (hysteresis-denied)", color="#8e6bbf", zorder=3,
                bottom=bot2, alpha=0.85)
    bot3 = [a + b for a, b in zip(bot2, peeked)]
    p4 = ax.bar(x, dropped,  bw, label="Dropped (evicted)",   color="#c0392b", zorder=3,
                bottom=bot3, alpha=0.7, hatch="///")

    totals = [a + b + c + dval for a, b, c, dval in zip(demoted, promoted, peeked, dropped)]
    for i, tot in enumerate(totals):
        ax.text(i, tot * 1.012, f"{tot/1e3:.0f}k", ha="center", va="bottom",
                fontsize=7.5, color="#222222")
    ax.set_ylim(0, max(totals) * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(tl, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Token Migration Events (cumulative)", fontsize=10)
    ax.set_title(
        "TieredKV Migration Profile per Task (Equal-VRAM Setting)",
        fontsize=10.5, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else str(int(v))))

    fig.tight_layout()
    savefig(fig, outdir, "fig6_migration_profile")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 & 2 — LaTeX accuracy tables
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy_table(data, tag, note):
    tasks   = data["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in data["config"]["methods"]]

    col_spec = "l" + "r" * len(methods)
    header   = " & ".join(
        [r"\textbf{Task}"] + [r"\textbf{" + METHOD_LABELS_PLAIN[m].replace("$", "") + "}" for m in methods]
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{LongBench accuracy (F1/ROUGE-L $\times$ 100). " + note + "}",
        r"\label{tab:accuracy_" + tag.lower().replace(" ", "_") + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    for t in tasks:
        scores = [get_score(data, t, m) for m in methods]
        formatted = bold_best(scores)
        row = TASK_LABELS_LATEX.get(t, t)
        row += " & " + " & ".join(formatted) + r" \\"
        lines.append(row)

    lines.append(r"\midrule")
    avgs = []
    for m in methods:
        sc_list = [get_score(data, t, m) for t in tasks]
        avgs.append(np.mean([s for s in sc_list if s is not None]))
    formatted_avg = bold_best(avgs)
    lines.append(r"\textbf{Average} & " + " & ".join(formatted_avg) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def table1_accuracy_equivram(data_ev, outdir):
    print("\n[Table 1] Accuracy (equal-VRAM)…")
    lines = _accuracy_table(
        data_ev,
        tag="equivram",
        note=r"Equal-VRAM setting: TieredKV uses VRAM=1024 + STT-RAM=2048; "
             r"baselines at budget=1024. Bold = best per task."
    )
    write_tex(lines, outdir, "table1_accuracy_equivram")
    print("\n".join(lines))


def table2_accuracy_fair(data_fair, outdir):
    print("\n[Table 2] Accuracy (fair / iso-VRAM)…")
    lines = _accuracy_table(
        data_fair,
        tag="fair",
        note=r"Fair (iso-VRAM) setting: all methods use identical VRAM budget (992 tokens). "
             r"Bold = best per task."
    )
    write_tex(lines, outdir, "table2_accuracy_fair")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 — Efficiency table (latency + energy + write savings, fair setting)
# ─────────────────────────────────────────────────────────────────────────────

def table3_efficiency(data_fair, outdir):
    print("\n[Table 3] Efficiency metrics (fair setting)…")
    tasks   = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in data_fair["config"]["methods"]]

    # Aggregate: total across tasks
    def total_stat(m, key, scale=1.0):
        vals = [get_stat(data_fair, t, m, key) for t in tasks]
        filtered = [v * scale for v in vals if v is not None]
        return sum(filtered) if filtered else None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Modeled efficiency metrics (fair/iso-VRAM setting, summed over all tasks). "
        r"Latency and energy are analytical estimates from the tier bandwidth model. "
        r"Bold = best (lowest) per metric. TieredKV uniquely tracks write savings.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Total Latency (s)} & \textbf{Total Energy (J)} & \textbf{Write Savings (\%)} \\",
        r"\midrule",
    ]

    lats    = []
    energys = []
    wsavs   = []
    for m in methods:
        lat  = total_stat(m, "modeled_latency_us", scale=1e-6)  # us -> s
        eng  = total_stat(m, "modeled_energy_nj",  scale=1e-9)  # nJ -> J
        # write savings only for TieredKV
        if m == "tieredkv":
            ws_list = []
            for t in tasks:
                pw = get_stat(data_fair, t, m, "paid_writes") or 0
                ws = get_stat(data_fair, t, m, "writes_saved") or 0
                total = pw + ws
                if total > 0:
                    ws_list.append(100.0 * ws / total)
            ws_val = np.mean(ws_list) if ws_list else None
        else:
            ws_val = None
        lats.append(lat)
        energys.append(eng)
        wsavs.append(ws_val)

    fmt_lat  = bold_best_low(lats,    fmt="{:.2f}")
    fmt_eng  = bold_best_low(energys, fmt="{:.2f}")

    for i, m in enumerate(methods):
        ws_str = f"{wsavs[i]:.1f}" if wsavs[i] is not None else "--"
        row = (
            METHOD_LABELS_PLAIN[m].replace("$", "").replace("_", r"\_")
            + " & " + fmt_lat[i]
            + " & " + fmt_eng[i]
            + " & " + ws_str
            + r" \\"
        )
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(lines, outdir, "table3_efficiency")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Table 4 — STT ablation table
# ─────────────────────────────────────────────────────────────────────────────

def table4_stt_ablation(sweep_paths, outdir):
    print("\n[Table 4] STT ablation table…")
    if not sweep_paths:
        print("  ⚠ No sweep files; skipping Table 4.")
        return

    rows = []
    for path in sorted(sweep_paths):
        m = re.search(r"(\d+)", os.path.basename(path))
        if not m:
            continue
        stt = int(m.group(1))
        d = load_json(path)
        tasks_in = list(d["results"].keys())
        scores, lats, wsavings = [], [], []
        for t in tasks_in:
            td = d["results"][t].get("tieredkv", {})
            sc = td.get("score")
            cs = td.get("cache_stats", {})
            lt = cs.get("modeled_latency_us")
            ws = cs.get("write_savings_pct")
            if sc is not None:
                scores.append(sc)
            if lt is not None:
                lats.append(lt / 1e6)  # us -> s
            if ws is not None:
                wsavings.append(ws)
        rows.append((
            stt,
            np.mean(scores) if scores else None,
            np.mean(lats)   if lats   else None,
            np.mean(wsavings) if wsavings else None,
        ))
    rows.sort()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{TieredKV ablation: STT-RAM budget vs. accuracy and efficiency "
        r"(5 short tasks, 25 samples; VRAM fixed at 1024; sweep at frac=1.0, "
        r"so write-savings read low — headline 83\% comes from frac=0.75).}",
        r"\label{tab:stt_ablation}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"\textbf{STT Budget} & \textbf{Avg F1} & \textbf{Avg Latency (s)} & \textbf{Write Savings (\%)} \\",
        r"\midrule",
    ]
    accs  = [r[1] for r in rows]
    lats  = [r[2] for r in rows]
    bests = max([v for v in accs if v is not None])

    for (stt, acc, lat, ws) in rows:
        acc_str = (r"\textbf{" + f"{acc:.2f}" + r"}" if acc is not None and abs(acc - bests) < 0.005
                   else f"{acc:.2f}" if acc is not None else "--")
        lat_str = f"{lat:.3f}" if lat is not None else "--"
        ws_str  = f"{ws:.1f}"  if ws  is not None else "--"
        lines.append(f"{stt} & {acc_str} & {lat_str} & {ws_str}" + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(lines, outdir, "table4_stt_ablation")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: print a clean ASCII summary table to stdout
# ─────────────────────────────────────────────────────────────────────────────

def print_ascii_summary(data_ev, data_fair):
    print("\n" + "═" * 80)
    print("  HEADLINE RESULTS SUMMARY")
    print("═" * 80)

    for data, label in [(data_ev, "Equal-VRAM"), (data_fair, "Fair / Iso-VRAM")]:
        tasks   = data["config"]["tasks"]
        methods = [m for m in METHOD_ORDER if m in data["config"]["methods"]]
        col_w   = 15

        print(f"\n── {label} ──")
        hdr = f"{'Task':<20}" + "".join(f"{METHOD_LABELS_PLAIN[m]:>{col_w}}" for m in methods)
        print(hdr)
        print("─" * len(hdr))
        for t in tasks:
            row = f"{TASK_LABELS.get(t, t):<20}"
            scores = [get_score(data, t, m) for m in methods]
            for sc in scores:
                row += f"{'--':>{col_w}}" if sc is None else f"{sc:>{col_w}.2f}"
            print(row)
        print("─" * len(hdr))
        avg_row = f"{'Average':<20}"
        for m in methods:
            sc_list = [get_score(data, t, m) for t in tasks]
            avg = np.mean([s for s in sc_list if s is not None]) if any(s is not None for s in sc_list) else None
            avg_row += f"{'--':>{col_w}}" if avg is None else f"{avg:>{col_w}.2f}"
        print(avg_row)
    print("═" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--equal-vram", required=True,
                    help="Path to final_equal_vram.json")
    ap.add_argument("--fair", required=True,
                    help="Path to final_equal_attended.json")
    ap.add_argument("--sweep-glob", default="results/clean_sweep_stt/stt*.json",
                    help="Glob for STT sweep JSONs")
    ap.add_argument("--outdir", default="figures/",
                    help="Output directory for figures")
    ap.add_argument("--tex-outdir", default="results/tables/",
                    help="Output directory for LaTeX table .tex files")
    args = ap.parse_args()

    print(f"\nLoading equal-VRAM data from: {args.equal_vram}")
    data_ev = load_json(args.equal_vram)

    print(f"Loading fair data from:        {args.fair}")
    data_fair = load_json(args.fair)

    sweep_paths = sorted(glob.glob(args.sweep_glob))
    print(f"Sweep files found:             {len(sweep_paths)}")

    print_ascii_summary(data_ev, data_fair)

    # ── Figures ──────────────────────────────────────────────────────────────
    # NOTE: fig4 (old accuracy-vs-latency scatter) is retired — its 10 numbers
    # live in the master table and Fig 4's slot now holds the decode-scaling
    # curves (plot_figures_v2.py: fig4_decode_scaling). Do not re-add the call.
    # NOTE 2: fig5 here is retired too — it mixed aggregate latency (ms) and
    # write-savings (%) on one axis and autoscaled noise into drama. Fig 5's
    # slot now holds twin-axis F1 + savings with honest ranges
    # (plot_figures_v2.py: fig5_clean). Do not re-add the call.
    fig1_accuracy_equivram(data_ev,   args.outdir)
    fig2_latency(data_fair,           args.outdir)
    fig3_write_savings(data_fair, data_ev, sweep_paths, args.outdir)
    fig6_migration_profile(data_ev,   args.outdir)
    fig7_accuracy_fair(data_fair,     args.outdir)

    # ── Tables ───────────────────────────────────────────────────────────────
    table1_accuracy_equivram(data_ev,   args.tex_outdir)
    table2_accuracy_fair(data_fair,     args.tex_outdir)
    table3_efficiency(data_fair,        args.tex_outdir)
    table4_stt_ablation(sweep_paths,    args.tex_outdir)

    print(f"\n{'═'*60}")
    print(f"  All figures saved to:  {args.outdir}")
    print(f"  All tables saved to:   {args.tex_outdir}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
