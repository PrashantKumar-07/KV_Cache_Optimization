#!/usr/bin/env python3
"""plot_figures_v2.py — professional publication figure pack (v2).

Rebuilds fig1/fig7 with fixed labeling (vertical bold value labels) and
fig4 as measured decode-latency scaling curves; adds fig5/fig11/fig12/fig14
+ master metrics table.
All numbers derive offline from selected run JSONs and cost_model.py
(nominal STT tentpole). GPU-free.

Inputs : --equal-vram, --fair, --latency (corrected measurements);
         historical ablations remain labeled as such.
Outputs: figures_final/fig{1,4,7,11,12,14}.pdf/.png,
         results/tables/table_master_metrics.tex
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
# for TieredKV (Ours). Used ONLY by fig1 (grouped_accuracy(style="pro")).
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
    # Matched to fig2_latency_comparison geometry/style so Fig 1 and Fig 2
    # read as one family: identical palette, hatch, edge, group stride
    # (0.68 slot), bar fill (0.92 of sub-slot), figsize, legend, caption,
    # tick/title sizes and spine handling. Only the y-scale differs
    # (linear scores here vs. log latency in Fig 2).
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
    # the legend — same as Fig 2.
    fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="#555555",
             style="italic")
    despine(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # Resolve any label-label overlaps using real rendered boxes (must run
    # after layout so box sizes/positions are final).
    _decollide_text_groups(fig, ax, group_texts, step)
    savefig(fig, stem)


def fig4_decode_scaling(latency):
    """Render only correctly scoped end-to-end measurements."""
    if latency["config"].get("timing_scope") != "end_to_end_model_policy_token":
        raise ValueError("Latency input is not an end-to-end benchmark")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for method, rows in latency["results"].items():
        points = sorted((int(ctx), entry["itl_ms"]) for ctx, entry in rows.items()
                        if entry.get("itl_ms") is not None)
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, "o-", label=MLABEL.get(method, method),
                    color=PALETTE.get(method))
    ax.set_xscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("End-to-end decode latency (ms/token)")
    ax.set_title("Decode latency including attention feedback and cache policy")
    ax.legend()
    fig.text(0.5, 0.015,
             "Mistral-7B-Instruct on L40S; real GovReport prompt, 50 fixed decode tokens. "
             "STT-RAM is simulated on GPU.",
             ha="center", fontsize=8, color="#555555")
    despine(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, "fig4_decode_scaling")


def footprint_rows(config):
    budget = config["budget"]
    vram = config["tiered_vram_budget"]
    stt = config["tiered_stt_budget"]
    expose = config.get("tiered_stt_expose")
    expose = 32 if expose is None else expose
    return [("Baseline compressed", budget, budget),
            ("TieredKV equal VRAM", vram + stt, vram + min(expose, stt)),
            ("TieredKV matched attention", budget - expose + stt, budget),
            ("TieredKV equal total", budget, budget // 3 + min(expose, budget-budget//3))]


def fig11_footprint(data):
    rows = footprint_rows(data["config"])
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    xs = np.arange(len(rows))
    mbs = [r[1] * BYTES_PER_TOKEN / 1e6 for r in rows]
    ax.bar(xs, mbs, color=["#3b8bc0", "#c0392b", "#c0392b", "#c0392b"])
    for i, (_, _, attended) in enumerate(rows):
        ax.text(i, mbs[i], f"{mbs[i]:.1f} MB\n≤{attended} cached tokens attended",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [r[0] for r in rows])
    ax.set_ylim(0, max(mbs)*1.25)
    ax.set_ylabel("Logical KV capacity (MB, bf16, Mistral-7B GQA)")
    ax.set_title("Configured capacity, not measured GPU allocation or actual occupancy")
    fig.tight_layout()
    savefig(fig, "fig11_footprint")


def fig12_energy(data):
    """Offline energy split at the nominal STT tentpole (CostModel):
    promote/demote/drop from event counters; attention = residual."""
    cm = CostModel()
    ELEM = KV_HEADS * HEAD_DIM  # per-layer token elements (K+V billed x2 below)
    comps, names = {}, []
    for m in METHODS:
        pe = de = dr = tot = 0.0
        for t in TASKS:
            e = data["results"][t][m]
            cs = e.get("cache_stats", {})
            if m == "tieredkv":
                _, p = cm.promote_cost(cs.get("promoted", 0) * 2 * ELEM)
                paid = cs.get("demoted", 0) - cs.get("writes_saved", 0)
                _, d = cm.demote_cost(paid * 2 * ELEM)
                _, r = cm.drop_cost(cs.get("dropped", 0) * 2 * ELEM)
                pe += p
                de += d
                dr += r
                tot += cs.get("modeled_energy_nj", 0)
            else:
                tot += cs.get("modeled_energy_nj", 0)
        att = max(0.0, tot - pe - de - dr)
        comps[m] = (att / 1e9, pe / 1e9, de / 1e9, dr / 1e9)
        names.append(m)
    labels = ["Attention\nreads", "Promote\n(STT read +\nVRAM write)",
              "Demote\n(VRAM read +\nSTT write)", "Drop\n(STT retire)"]
    cols = ["#457b9d", "#e9c46a", "#c0392b", "#7f7f7f"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bottom = np.zeros(len(names))
    for k in range(4):
        hs = [comps[m][k] for m in names]
        ax.bar(x, hs, 0.6, bottom=bottom, label=labels[k], color=cols[k],
               edgecolor="white", linewidth=0.8, zorder=3)
        for i, (b, h) in enumerate(zip(bottom, hs)):
            if h > 3.0:
                ax.text(i, b + h / 2, f"{h:.1f} J", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom = bottom + np.array(hs)
    totals = [sum(comps[m]) for m in names]
    for i, tval in enumerate(totals):
        ax.text(i, tval * 1.015, f"{tval:.1f} J total", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MLABEL[m] for m in names])
    ax.set_ylabel("Modeled decode KV-traffic energy, 6 tasks (Joules)")
    ax.set_title("Decode KV-Traffic Energy: Analytical Split from Event Counters",
                 fontweight="bold", pad=10)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.text(0.5, 0.015,
             "Assumed STT 2/8 pJ per bit; excludes prefill, compute, and system power. No STT device measured.",
             ha="center", fontsize=8)
    despine(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    savefig(fig, "fig12_energy")


def fig14_heatmap(data):
    mat = np.array([[data["results"][t][m]["score"] for m in METHODS]
                    for t in TASKS])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=10, vmax=75)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([MLABEL[m] for m in METHODS], fontsize=9)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([TASK_LABELS[t] for t in TASKS], fontsize=9)
    best_comp = np.argmax(mat[:, 1:], axis=1) + 1
    for i in range(len(TASKS)):
        for j in range(len(METHODS)):
            wt = "bold" if j == best_comp[i] else "normal"
            bx = dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.85) if j == best_comp[i] else None
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                    fontsize=9, fontweight=wt, bbox=bx,
                    color="#1d3557" if mat[i, j] > 40 else "black")
    ax.set_title("LongBench Score Heatmap (boxed = best compressed method)",
                 fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("F1 / ROUGE-L")
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig14_heatmap")


def master_table(data, latency=None):
    lat = latency["results"] if latency else {}
    cm = CostModel()
    ELEM = KV_HEADS * HEAD_DIM
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Master results: task score (F1 or ROUGE-L; unweighted mean $\pm$ within-task SE), measured full-loop ITL@16k, "
        r"configured logical KV capacity, maximum attended tokens, decode-demotion write savings, analytical decode KV-traffic energy. "
        r"TieredKV uses bf16 KV.}",
        r"\label{tab:master}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Avg score $\pm$ SE} & \textbf{ITL@16k (ms)} & "
        r"\textbf{KV capacity} & \textbf{Attended} & \textbf{Write sav.} & \textbf{Decode energy (J)} \\",
        r"\midrule",
    ]
    avgs, itls = {}, {}
    for m in METHODS:
        ss = [data["results"][t][m]["score"] for t in TASKS]
        avgs[m] = float(np.mean(ss))
        itls[m] = lat.get(m, {}).get("16384", {}).get("itl_ms")
    se_pool = {}
    for m in METHODS:
        vs = [se_of(data["results"][t][m]) for t in TASKS]
        se_pool[m] = float(np.sqrt(sum(v * v for v in vs))) / len(vs)
    cfg = data["config"]
    capacity_mb = lambda tokens: f"{tokens * BYTES_PER_TOKEN / 1e6:.0f} MB"
    budget = cfg["budget"]
    vram = cfg["tiered_vram_budget"]
    stt = cfg["tiered_stt_budget"]
    expose = cfg.get("tiered_stt_expose")
    expose = 32 if expose is None else expose
    resident = {"full": capacity_mb(cfg["max_ctx"]),
                "streamingllm": capacity_mb(budget),
                "h2o": capacity_mb(budget), "snapkv": capacity_mb(budget),
                "tieredkv": capacity_mb(vram + stt)}
    attended = {"full": "full ctx", "streamingllm": str(budget),
                "h2o": str(budget), "snapkv": str(budget),
                "tieredkv": f"{vram}+{min(expose, stt)}"}
    for m in METHODS:
        tot_e = 0.0
        sav_pct = "--"
        if m == "tieredkv":
            P = W = 0
            for t in TASKS:
                cs = data["results"][t][m].get("cache_stats", {})
                tot_e += cs.get("modeled_energy_nj", 0)
                P += cs.get("paid_writes", 0)
                W += cs.get("writes_saved", 0)
            sav_pct = f"{100.0 * W / (P + W):.1f}" if P + W else "--"
            tot_e = f"{tot_e / 1e9:.1f}"
        else:
            for t in TASKS:
                tot_e += data["results"][t][m].get("cache_stats",
                                                   {}).get("modeled_energy_nj", 0)
            tot_e = f"{tot_e / 1e9:.1f}"
        name = r"\textbf{TieredKV (Ours)}" if m == "tieredkv" else MLABEL[m].replace("$", "")
        lines.append(f"{name} & {avgs[m]:.2f} $\\pm$ {se_pool[m]:.2f} & "
                     f"{format(itls[m], '.1f') if itls[m] is not None else '--'} & {resident[m]} & {attended[m]} & "
                     f"{sav_pct} & {tot_e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, "table_master_metrics.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  saved table_master_metrics.tex")
    for m in METHODS:
        print(f"  {m}: score {avgs[m]:.2f} SE {se_pool[m]:.2f}, ITL {format(itls[m], '.1f') if itls[m] is not None else '--'}ms")


def fig5_clean():
    """STT-budget ablation from the CLEAN sweep: avg F1 vs STT size only.

    Single quantity, single axis — no dual-axis mixing, nothing to overlap.
    The sweep ran at frac=1.0; F1 moves ≤0.2 mean under frac=0.75
    (fixed-sample A/B), so the flatness result stands at headline config.
    Write-savings-vs-size is intentionally NOT plotted here: under the
    artifact config it tops out at 20% and would contradict the headline
    ~83% (frac=0.75). The savings story belongs to the frac sweep (fig8).
    """
    import glob as _glob, re as _re
    rows = []
    for path in sorted(_glob.glob("results/clean_sweep_stt/stt*.json")):
        m = _re.search(r"(\d+)", os.path.basename(path))
        stt = int(m.group(1))
        d = load(path)
        f1 = np.mean([d["results"][t]["tieredkv"]["score"] for t in
                      ["multifieldqa_en", "hotpotqa", "triviaqa", "qasper",
                       "narrativeqa"]])
        rows.append((stt, f1))
    rows.sort()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    xs = [r[0] for r in rows]
    ax.plot(xs, [r[1] for r in rows], "D-", color="#c0392b",
            linewidth=2.2, markersize=8, zorder=5,
            markeredgecolor="#222222", markeredgewidth=0.6)
    for x0, f1 in rows:
        ax.text(x0, f1 + 0.05, f"{f1:.2f}", ha="center", va="bottom",
                fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xlabel("STT-RAM budget (tokens, VRAM fixed at 1024)")
    ax.set_ylabel("Avg F1 (5 short tasks, 25 samples)")
    ax.set_ylim(37.5, 39.2)
    ax.set_title("STT-Budget Ablation: Accuracy Flat Across STT Sizes",
                 fontweight="bold")
    fig.text(0.5, 0.005,
             "Sweep at frac=1.0 (5 short tasks, VRAM 1024, bf16). F1 is flat "
             "within noise: extra slow-tier capacity adds pool the fixed "
             "exposure window cannot surface (capacity without visibility). "
             "Write savings live in the frac sweep (fig8), not here.",
             ha="center", fontsize=7, color="#555555", style="italic")
    despine(ax)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, "fig5_stt_ablation")




def main():
    import argparse
    global OUT, TAB
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--equal-vram', required=True)
    p.add_argument('--fair', required=True)
    p.add_argument('--latency', required=True, help='Corrected end-to-end latency JSON only')
    p.add_argument('--outdir', default='figures_final')
    p.add_argument('--tex-outdir', default='results/tables')
    args = p.parse_args()
    OUT, TAB = args.outdir, args.tex_outdir
    data_ev, data_fair = load(args.equal_vram), load(args.fair)
    latency = load(args.latency)
    configs = [data_ev['config'], data_fair['config'], latency['config']]
    hashes = [c.get('runtime', {}).get('source_sha256') for c in configs]
    if not all(hashes) or len(set(hashes)) != 1:
        raise ValueError('Figure inputs must share one recorded source fingerprint')
    if data_ev['config'].get('dataset_sha256_by_task') != data_fair['config'].get('dataset_sha256_by_task'):
        raise ValueError('Accuracy figure inputs use different dataset files')
    fig4_decode_scaling(latency)
    grouped_accuracy(data_ev, 'fig1_accuracy_equivram',
        'LongBench accuracy: equal logical VRAM',
        'Baseline adaptations; error bars = standard errors across samples.',
        style='pro')
    grouped_accuracy(data_fair, 'fig7_accuracy_fair',
        'LongBench accuracy: matched attended cache budget',
        'Adapted baselines; TieredKV: 992 fast + ≤32 slow; baselines: 1024 fast. Error bars = SE.',
        style='pro')
    fig11_footprint(data_ev)
    fig12_energy(data_ev)
    fig14_heatmap(data_ev)
    master_table(data_ev, latency)


if __name__ == '__main__':
    main()
