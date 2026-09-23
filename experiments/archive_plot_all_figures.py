#!/usr/bin/env python3
"""plot_all_figures.py — Publication-quality figures from real measured data.

Style: NeurIPS / ICLR / ICML top-tier paper standard
  - Light, clean, easily distinguishable color palette
  - High-clarity subplots and inset zoom boxes (like Quest & StreamingLLM papers)
  - Non-overlapping labels, clear Pareto frontier and annotated callouts
"""

import os, sys, re, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Publication style ────────────────────────────────────────────────────────
# Light, vivid, easily distinguishable color palette
COLORS = {
    "full":         "#2CA02C",   # Soft Emerald Green (Baseline)
    "streamingllm": "#1F77B4",   # Soft Bright Blue
    "h2o":          "#FF7F0E",   # Soft Bright Orange
    "snapkv":       "#9467BD",   # Soft Bright Purple
    "tieredkv":     "#D62728",   # Clean Crimson Red (TieredKV / Ours)
}

LABELS = {
    "full":         "Full Cache",
    "streamingllm": "StreamingLLM",
    "h2o":          r"H$_2$O",
    "snapkv":       "SnapKV",
    "tieredkv":     "TieredKV (Ours)",
}

MARKERS = {
    "full":         "o",
    "streamingllm": "o",
    "h2o":          "s",
    "snapkv":       "^",
    "tieredkv":     "D",
}

LINE_STYLES = {
    "full":         "--",
    "streamingllm": "-",
    "h2o":          "-",
    "snapkv":       "-",
    "tieredkv":     "-",
}

TASK_LABELS = {
    "multifieldqa_en": "MultiFieldQA",
    "hotpotqa":        "HotpotQA",
    "triviaqa":        "TriviaQA",
    "qasper":          "Qasper",
    "narrativeqa":     "NarrativeQA",
    "gov_report":      "GovReport",
}

METHOD_ORDER = ["full", "streamingllm", "h2o", "snapkv", "tieredkv"]


def setup_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Times New Roman", "Palatino"],
        "font.size":          8.5,
        "axes.titlesize":     9,
        "axes.labelsize":     8.5,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    7.5,
        "legend.frameon":     True,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "#e0e0e0",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         "#e8e8e8",
        "grid.alpha":         0.7,
        "grid.linewidth":     0.5,
        "lines.linewidth":    1.5,
        "lines.markersize":   4.5,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.03,
    })


def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p)
        print(f"  ✓  {p}")
    plt.close(fig)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_score(data, task, method):
    """Extract F1/ROUGE score from a results JSON."""
    r = data.get("results", {}).get(task, {}).get(method, {})
    return r.get("score")


def get_stat(data, task, method, stat_key):
    """Extract a cache_stat value."""
    r = data.get("results", {}).get(task, {}).get(method, {})
    cs = r.get("cache_stats", {})
    return cs.get(stat_key)


# ─── Figure 1: Accuracy Bar Chart (Equal-VRAM) ────────────────────────────────
def fig_accuracy_equivram(data_ev, outdir):
    print("\n[Fig 1] Accuracy bar chart (equal-VRAM)…")
    tasks = data_ev["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in list(data_ev["results"].values())[0]]

    n_tasks = len(tasks)
    n_methods = len(methods)
    x = np.arange(n_tasks)
    bw = 0.75 / n_methods

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    all_scores_list = []
    for i, m in enumerate(methods):
        scores = [get_score(data_ev, t, m) or 0.0 for t in tasks]
        all_scores_list.extend(scores)
        offset = (i - n_methods / 2 + 0.5) * bw
        ax.bar(x + offset, scores, bw, label=LABELS[m],
               color=COLORS[m], alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=12, ha="right")
    ax.set_ylabel("Score (F1 / ROUGE-L)")
    ax.set_title("LongBench Benchmark Accuracy — Equal-VRAM Setting")
    if all_scores_list:
        ax.set_ylim(0, max(all_scores_list) * 1.18)
    ax.legend(loc="upper right", ncol=3, framealpha=0.9, fontsize=8)
    fig.tight_layout()
    savefig(fig, outdir, "fig1_accuracy_equivram")


# ─── Figure 2: Accuracy Bar Chart (Fair / Iso-VRAM) ───────────────────────────
def fig_accuracy_fair(data_fair, outdir):
    print("\n[Fig 2] Accuracy bar chart (fair / iso-VRAM)…")
    tasks = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in list(data_fair["results"].values())[0]]

    n_tasks = len(tasks)
    n_methods = len(methods)
    x = np.arange(n_tasks)
    bw = 0.75 / n_methods

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    all_scores_list = []
    for i, m in enumerate(methods):
        scores = [get_score(data_fair, t, m) or 0.0 for t in tasks]
        all_scores_list.extend(scores)
        offset = (i - n_methods / 2 + 0.5) * bw
        ax.bar(x + offset, scores, bw, label=LABELS[m],
               color=COLORS[m], alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=12, ha="right")
    ax.set_ylabel("Score (F1 / ROUGE-L)")
    ax.set_title("LongBench Benchmark Accuracy — Iso-VRAM Setting")
    if all_scores_list:
        ax.set_ylim(0, max(all_scores_list) * 1.18)
    ax.legend(loc="upper right", ncol=3, framealpha=0.9, fontsize=8)
    fig.tight_layout()
    savefig(fig, outdir, "fig2_accuracy_fair")


# ─── Figure 3: F1 vs. KV Cache Budget (6-panel grid like Quest paper) ────────
def fig_f1_vs_budget(budget_data_list, data_fair, outdir):
    """6-panel figure: one subplot per task, lines per method vs KV budget."""
    print("\n[Fig 3] F1 vs. KV Cache Budget (6-panel grid)…")
    tasks = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m != "full"]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    axes = axes.flatten()

    for ax_idx, task in enumerate(tasks):
        ax = axes[ax_idx]
        task_title = TASK_LABELS.get(task, task)
        ax.set_title(task_title, fontweight="bold", fontsize=9)

        # Full Cache score (horizontal dashed line)
        full_score = None
        for _, bdata in budget_data_list:
            s = get_score(bdata, task, "full")
            if s is not None:
                full_score = s
                break
        if full_score is None:
            full_score = get_score(data_fair, task, "full")

        if full_score is not None:
            ax.axhline(full_score, color=COLORS["full"], linestyle="--",
                       linewidth=1.2, label=LABELS["full"], zorder=4)

        # Lines for eviction methods
        for m in methods:
            xs, ys = [], []
            for budget, bdata in budget_data_list:
                s = get_score(bdata, task, m)
                if s is not None:
                    xs.append(budget)
                    ys.append(s)
            if not xs:
                s = get_score(data_fair, task, m)
                if s is not None:
                    xs, ys = [1024], [s]
            if xs:
                lw = 1.8 if m == "tieredkv" else 1.2
                zorder = 5 if m == "tieredkv" else 3
                ax.plot(xs, ys, marker=MARKERS[m], label=LABELS[m],
                        color=COLORS[m], linewidth=lw, linestyle=LINE_STYLES[m],
                        zorder=zorder, markersize=4)

        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 512, 1024, 2048, 4096])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        if ax_idx >= 3:
            ax.set_xlabel("KV Cache Budget", fontsize=8)
        if ax_idx % 3 == 0:
            ax.set_ylabel("F1 / ROUGE Score", fontsize=8)

    # Global legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=5, frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, outdir, "fig3_f1_vs_budget")


# ─── Figure 4: PPL vs Context Length (With Inset Zoom like paper image) ───────
def fig_ppl_vs_context(ppl_data, outdir):
    print("\n[Fig 4] PPL vs. Context Length (with Inset Zoom)…")
    if not ppl_data:
        print("  ⚠ No PPL data — skipping")
        return

    results = ppl_data.get("results", {})
    methods_present = [m for m in METHOD_ORDER if m in results]
    if not methods_present:
        print("  ⚠ No PPL methods found — skipping")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    # Inset axes positioned cleanly on upper right without touching title
    axins = ax.inset_axes([0.52, 0.48, 0.44, 0.44])

    for m in methods_present:
        curve = results[m].get("ppl_curve", {})
        xs = sorted([int(k) for k in curve.keys()])
        ys = [curve[str(x)] for x in xs]

        lw = 2.0 if m == "tieredkv" else 1.3
        zorder = 5 if m == "tieredkv" else 3

        ax.plot(xs, ys, marker=MARKERS[m], label=LABELS[m],
                color=COLORS[m], linewidth=lw, linestyle=LINE_STYLES[m],
                zorder=zorder, markersize=4.5)

        # Plot in inset zoom
        axins.plot(xs, ys, marker=MARKERS[m], color=COLORS[m],
                   linewidth=lw, linestyle=LINE_STYLES[m], markersize=3.5)

    ax.set_xlabel("Context / Input Length (tokens)", fontsize=9.5)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=9.5)
    ax.set_title("Language Modeling Perplexity vs. Context Length", fontsize=10.5, pad=14, fontweight="bold")
    ax.set_ylim(3.8, 5.5)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.0)

    # Set zoom window for inset: focus on context lengths >= 4096
    all_xs = sorted([int(k) for k in results[methods_present[0]]["ppl_curve"].keys()])
    high_xs = [x for x in all_xs if x >= 4096]
    if high_xs:
        x1, x2 = min(high_xs) - 200, max(high_xs) + 200
        # Calculate y-limits for zoomed area
        zoom_ys = []
        for m in methods_present:
            c = results[m]["ppl_curve"]
            zoom_ys.extend([c[str(x)] for x in high_xs if str(x) in c])
        if zoom_ys:
            y1, y2 = min(zoom_ys) - 0.1, max(zoom_ys) + 0.15
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.tick_params(axis='both', which='major', labelsize=6.5)
            axins.grid(True, alpha=0.4, linestyle=":")
            ax.indicate_inset_zoom(axins, edgecolor="#777777", alpha=0.5)

    fig.tight_layout()
    savefig(fig, outdir, "fig4_ppl_vs_context")


# ─── Figure 5: Wall-Clock Latency Profile ─────────────────────────────────────
def fig_latency_wallclock(lat_data, outdir):
    print("\n[Fig 5] Wall-clock latency (TTFT + ITL)…")
    if not lat_data:
        print("  ⚠ No latency data — skipping")
        return

    results = lat_data.get("results", {})
    methods_present = [m for m in METHOD_ORDER if m in results]
    if not methods_present:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    for m in methods_present:
        mdata = results[m]
        ctx_lens = sorted([int(k) for k in mdata.keys()])
        ttft_vals = [mdata[str(c)]["ttft_ms"] for c in ctx_lens]
        itl_vals  = [mdata[str(c)]["itl_ms"]  for c in ctx_lens]

        lw = 1.8 if m == "tieredkv" else 1.2
        ax1.plot(ctx_lens, ttft_vals, marker=MARKERS[m], label=LABELS[m],
                 color=COLORS[m], linewidth=lw, linestyle=LINE_STYLES[m])
        ax2.plot(ctx_lens, itl_vals,  marker=MARKERS[m], label=LABELS[m],
                 color=COLORS[m], linewidth=lw, linestyle=LINE_STYLES[m])

    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("TTFT (Prefill Latency, ms)")
    ax1.set_title("(a) Time-To-First-Token (TTFT)")

    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("ITL (Decode Latency / Step, ms)")
    ax2.set_title("(b) Inter-Token Latency (ITL)")
    ax2.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    savefig(fig, outdir, "fig5_wallclock_latency")


# ─── Figure 6: Memory Footprint ───────────────────────────────────────────────
def fig_memory_footprint(lat_data, outdir):
    print("\n[Fig 6] KV cache memory footprint…")
    if not lat_data:
        return

    results = lat_data.get("results", {})
    methods_present = [m for m in METHOD_ORDER if m in results]
    if not methods_present:
        return

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for m in methods_present:
        mdata = results[m]
        ctx_lens = sorted([int(k) for k in mdata.keys()])
        mem_pts = [mdata[str(c)]["peak_memory_gb"] for c in ctx_lens]

        lw = 1.8 if m == "tieredkv" else 1.2
        ax.plot(ctx_lens, mem_pts, marker=MARKERS[m], label=LABELS[m],
                color=COLORS[m], linewidth=lw, linestyle=LINE_STYLES[m])

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Peak VRAM Memory (GB)")
    ax.set_title("Peak VRAM Footprint vs. Context Length")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    savefig(fig, outdir, "fig6_memory_footprint")


# ─── Figure 7: Modeled Decode Latency ─────────────────────────────────────────
def fig_modeled_latency(data_fair, outdir):
    print("\n[Fig 7] Modeled decode latency…")
    tasks = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER]

    n_tasks = len(tasks)
    n_methods = len(methods)
    x = np.arange(n_tasks)
    bw = 0.75 / n_methods

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for i, m in enumerate(methods):
        lats_ms = []
        for t in tasks:
            cs = data_fair.get("results", {}).get(t, {}).get(m, {}).get("cache_stats", {})
            lat_us = cs.get("modeled_latency_us", 0) or 0
            lats_ms.append(lat_us / 1000.0)

        offset = (i - n_methods / 2 + 0.5) * bw
        ax.bar(x + offset, lats_ms, bw, label=LABELS[m],
               color=COLORS[m], alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=12, ha="right")
    ax.set_ylabel("Modeled Latency per Token (ms)")
    ax.set_title("Analytical Hardware Cost Model: Decode Latency per Token")
    ax.legend(loc="upper right", ncol=3, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, outdir, "fig7_modeled_latency")


# ─── Figure 8: STT Budget Ablation ───────────────────────────────────────────
def fig_stt_ablation(sweep_paths, outdir):
    print("\n[Fig 8] STT budget ablation…")
    if not sweep_paths:
        return

    stt_budgets, avg_scores, avg_lats_ms, write_savs = [], [], [], []

    for path in sorted(sweep_paths):
        m = re.search(r"(\d+)", os.path.basename(path))
        if not m: continue
        stt = int(m.group(1))
        d = load_json(path)
        tasks = d["config"].get("tasks", [])

        scores, lats, ws_pcts = [], [], []
        for t in tasks:
            s = get_score(d, t, "tieredkv")
            if s is not None: scores.append(s)
            cs = d.get("results", {}).get(t, {}).get("tieredkv", {}).get("cache_stats", {})
            lat_us = cs.get("modeled_latency_us", 0) or 0
            if lat_us: lats.append(lat_us / 1000.0)
            pw = cs.get("paid_writes", 0) or 0
            ws = cs.get("writes_saved", 0) or 0
            if (pw + ws) > 0: ws_pcts.append(100.0 * ws / (pw + ws))

        stt_budgets.append(stt)
        avg_scores.append(np.mean(scores) if scores else 0)
        avg_lats_ms.append(np.mean(lats)  if lats   else 0)
        write_savs.append(np.mean(ws_pcts) if ws_pcts else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    ax1.plot(stt_budgets, avg_scores, marker="D", color=COLORS["tieredkv"], linewidth=1.8)
    ax1.set_xlabel("STT-RAM Budget (tokens)")
    ax1.set_ylabel("Avg. LongBench Score")
    ax1.set_title("(a) Accuracy vs. STT Budget")

    ax2.plot(stt_budgets, write_savs, marker="s", color="#3498DB", linewidth=1.8)
    ax2.set_xlabel("STT-RAM Budget (tokens)")
    ax2.set_ylabel("NVM Write Savings (%)")
    ax2.set_title("(b) Write Savings vs. STT Budget")

    fig.tight_layout()
    savefig(fig, outdir, "fig8_stt_ablation")


# ─── Figure 9: Migration Profile ──────────────────────────────────────────────
def fig_migration_profile(data_ev, outdir):
    print("\n[Fig 9] Migration profile (stacked bar)…")
    tasks = data_ev["config"]["tasks"]

    demoted  = [get_stat(data_ev, t, "tieredkv", "demoted")   or 0 for t in tasks]
    promoted = [get_stat(data_ev, t, "tieredkv", "promoted")  or 0 for t in tasks]
    peeked   = [get_stat(data_ev, t, "tieredkv", "promotions_deferred") or 0 for t in tasks]
    dropped  = [get_stat(data_ev, t, "tieredkv", "dropped")   or 0 for t in tasks]

    x = np.arange(len(tasks))
    tlabs = [TASK_LABELS.get(t, t) for t in tasks]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    b1 = ax.bar(x, demoted,  0.55, label="Demoted (STT→NVM)",  color="#2980B9", alpha=0.88)
    b2 = ax.bar(x, promoted, 0.55, bottom=demoted, label="Promoted (NVM→STT)", color="#27AE60", alpha=0.88)
    b3 = ax.bar(x, peeked,   0.55, bottom=np.array(demoted)+np.array(promoted), label="Peeked (Deferred)", color="#F39C12", alpha=0.88)
    b4 = ax.bar(x, dropped,  0.55, bottom=np.array(demoted)+np.array(promoted)+np.array(peeked), label="Evicted / Dropped", color="#E74C3C", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(tlabs, rotation=12, ha="right")
    ax.set_ylabel("Total Key-Value Blocks / Operations")
    ax.set_title("TieredKV Cache Traffic Breakdown per Task")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=7.5)
    fig.tight_layout()
    savefig(fig, outdir, "fig9_migration_profile")


# ─── Figure 10: NVM Endurance Multiplier (FIXED OVERLAPPING LABELS) ──────────
def fig_nvm_endurance(data_ev, data_fair, outdir):
    print("\n[Fig 10] NVM endurance multiplier (Fixed non-overlapping layout)…")
    tasks = data_fair["config"]["tasks"]

    mult_ev, mult_fair = [], []
    t_labels = []

    for t in tasks:
        dem_ev = get_stat(data_ev, t, "tieredkv", "demoted") or 0
        pw_ev  = get_stat(data_ev, t, "tieredkv", "paid_writes") or 0
        m_ev = (dem_ev / pw_ev) if pw_ev > 0 else 0.0
        mult_ev.append(round(m_ev, 2))

        dem_f = get_stat(data_fair, t, "tieredkv", "demoted") or 0
        pw_f  = get_stat(data_fair, t, "tieredkv", "paid_writes") or 0
        m_f = (dem_f / pw_f) if pw_f > 0 else 0.0
        mult_fair.append(round(m_f, 2))

        t_labels.append(TASK_LABELS.get(t, t))

    x = np.arange(len(tasks))
    bw = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    rects1 = ax.bar(x - bw/2, mult_ev, bw, label="Equal-VRAM Setting",
                    color="#9B59B6", alpha=0.88, edgecolor="white", linewidth=0.5)
    rects2 = ax.bar(x + bw/2, mult_fair, bw, label="Iso-VRAM Setting",
                    color="#EC7063", alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--", label="Baseline (1.0×)")

    # Value labels above bars
    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}×", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}×", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, rotation=12, ha="right", fontsize=8.5)
    ax.set_ylabel("Endurance Multiplier (Demotions / Paid Writes)", fontsize=8.5)
    ax.set_title("NVM Endurance Extension Multiplier by Task\n"
                 r"$\text{Endurance Multiplier} = \frac{\text{Demoted Blocks to NVM}}{\text{Direct NVM Writes}}$")
    ax.set_ylim(0, max(max(mult_ev), max(mult_fair)) * 1.25)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    savefig(fig, outdir, "fig10_nvm_endurance")


# ─── Figure 11: Accuracy vs. Efficiency Trade-off (2-panel Pareto plot) ──────
def fig_pareto_scatter(data_fair, outdir):
    print("\n[Fig 11] Accuracy vs. Efficiency Pareto Trade-off (2-panel layout)…")
    tasks = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER]

    avg_scores, avg_lats, avg_ws = {}, {}, {}
    for m in methods:
        scores, lats, ws_list = [], [], []
        for t in tasks:
            s = get_score(data_fair, t, m)
            if s is not None: scores.append(s)
            cs = data_fair.get("results", {}).get(t, {}).get(m, {}).get("cache_stats", {})
            lat_us = cs.get("modeled_latency_us", 0) or 0
            lats.append(lat_us / 1000.0)
            pw = cs.get("paid_writes", 0) or 0
            ws = cs.get("writes_saved", 0) or 0
            tot = pw + ws
            ws_list.append((100.0 * ws / tot) if tot > 0 else 0.0)

        avg_scores[m] = np.mean(scores) if scores else 0
        avg_lats[m]   = np.mean(lats) if lats else 0
        avg_ws[m]     = np.mean(ws_list) if ws_list else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # --- Subplot (a): Quality vs Latency ---
    for m in methods:
        size = 140 if m == "tieredkv" else 75
        marker = MARKERS[m]
        ax1.scatter(avg_lats[m], avg_scores[m], s=size, marker=marker,
                    color=COLORS[m], label=LABELS[m], zorder=5,
                    edgecolors="black", linewidths=0.6)

    # Offset annotations to prevent overlapping text
    offsets_a = {
        "full":         (-35, -16),
        "tieredkv":     (8, -4),
        "snapkv":       (8, 4),
        "h2o":          (8, -8),
        "streamingllm": (8, -6),
    }
    for m in methods:
        ox, oy = offsets_a.get(m, (8, 5))
        ax1.annotate(LABELS[m], (avg_lats[m], avg_scores[m]),
                     textcoords="offset points", xytext=(ox, oy),
                     fontsize=7.5, color=COLORS[m], fontweight="bold" if m=="tieredkv" else "normal")

    # Connect Pareto frontier points (Full -> TieredKV -> SnapKV -> H2O -> StreamingLLM)
    p_methods = ["streamingllm", "h2o", "snapkv", "tieredkv", "full"]
    p_lats = [avg_lats[m] for m in p_methods]
    p_scores = [avg_scores[m] for m in p_methods]
    ax1.plot(p_lats, p_scores, linestyle=":", color="#777777", linewidth=1.2, zorder=2, label="Trade-off Boundary")

    ax1.set_xlabel("Avg. Modeled Latency per Token (ms)", fontsize=8.5)
    ax1.set_ylabel("Avg. Benchmark Score (F1 / ROUGE)", fontsize=8.5)
    ax1.set_title("(a) Quality vs. Hardware Latency", fontsize=9)
    ax1.set_ylim(30, 48)
    ax1.set_xlim(0, 135)
    ax1.annotate("← Faster Latency | Higher Score →", xy=(0.03, 0.90), xycoords="axes fraction",
                 fontsize=6.5, color="#555555", bbox=dict(boxstyle="round,pad=0.3", fc="#f9f9f9", ec="#dddddd"))

    # --- Subplot (b): Quality vs Write Savings ---
    for m in methods:
        size = 140 if m == "tieredkv" else 75
        marker = MARKERS[m]
        ax2.scatter(avg_ws[m], avg_scores[m], s=size, marker=marker,
                    color=COLORS[m], label=LABELS[m], zorder=5,
                    edgecolors="black", linewidths=0.6)
        ox, oy = offsets_a.get(m, (8, 5))
        if m in ["full", "streamingllm", "h2o", "snapkv"]:
            # Shift x offset for 0% write savings to avoid y-axis overlap
            ox = 10
        ax2.annotate(LABELS[m], (avg_ws[m], avg_scores[m]),
                     textcoords="offset points", xytext=(ox, oy),
                     fontsize=7.5, color=COLORS[m], fontweight="bold" if m=="tieredkv" else "normal")

    ax2.set_xlabel("NVM Write Savings (%)", fontsize=8.5)
    ax2.set_ylabel("Avg. Benchmark Score (F1 / ROUGE)", fontsize=8.5)
    ax2.set_title("(b) Quality vs. Write Reduction", fontsize=9)
    ax2.set_ylim(30, 48)
    ax2.set_xlim(-2, 16)
    ax2.annotate("Ideal Zone\n(High Score & High Savings)", xy=(10.5, 38.5),
                 fontsize=7.0, color="#D62728", fontweight="bold", ha="center")

    fig.tight_layout()
    savefig(fig, outdir, "fig11_pareto_scatter")


# ─── LaTeX Tables ─────────────────────────────────────────────────────────────

def table_accuracy(data, setting_desc, tex_outdir, filename):
    tasks = data["config"]["tasks"]
    methods = [m for m in METHOD_ORDER if m in list(data["results"].values())[0]]

    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{LongBench accuracy comparison across methods ({setting_desc}). "
        "Bold = best baseline, \\textbf{\\underline{bold underline}} = overall best.}",
        f"\\label{{tab:{filename}}}",
        "\\begin{tabular}{l" + "r" * len(tasks) + "r}",
        "\\toprule",
        "\\textbf{Method} & " + " & ".join([f"\\textbf{{{TASK_LABELS.get(t,t)}}}" for t in tasks]) + " & \\textbf{Avg.} \\\\",
        "\\midrule",
    ]

    for m in methods:
        scores = [get_score(data, t, m) or 0.0 for t in tasks]
        avg_s = float(np.mean(scores))
        row_str = [LABELS[m]]
        for t, s in zip(tasks, scores):
            row_str.append(f"{s:.2f}")
        row_str.append(f"{avg_s:.2f}")
        lines.append(" & ".join(row_str) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    os.makedirs(tex_outdir, exist_ok=True)
    out = os.path.join(tex_outdir, f"{filename}.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓  {out}")


def table_efficiency(data_fair, tex_outdir):
    tasks = data_fair["config"]["tasks"]
    methods = [m for m in METHOD_ORDER]

    rows = []
    for m in methods:
        tot_lat, tot_eng = 0.0, 0.0
        tot_pw, tot_ws, tot_dem = 0, 0, 0
        for t in tasks:
            cs = data_fair.get("results", {}).get(t, {}).get(m, {}).get("cache_stats", {})
            tot_lat += cs.get("modeled_latency_us", 0) or 0
            tot_eng += cs.get("modeled_energy_nj", 0)  or 0
            tot_pw  += cs.get("paid_writes", 0)        or 0
            tot_ws  += cs.get("writes_saved", 0)       or 0
            tot_dem += cs.get("demoted", 0)            or 0

        tot_writes = tot_pw + tot_ws
        ws_pct = (100.0 * tot_ws / tot_writes) if tot_writes > 0 else None
        endurance = (tot_dem / tot_pw) if tot_pw > 0 else None

        rows.append({
            "method": m,
            "lat_s":  tot_lat / 1e6,
            "eng_j":  tot_eng / 1e9,
            "ws_pct": ws_pct,
            "endurance": endurance,
        })

    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Modeled efficiency (iso-VRAM setting, summed over tasks). "
        "Latency and energy from analytical hardware model.}",
        "\\label{tab:efficiency}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Latency (s)} & \\textbf{Energy (J)} & "
        "\\textbf{Write Sav. (\\%)} & \\textbf{NVM Endur.} \\\\",
        "\\midrule",
    ]
    for r in rows:
        m = r["method"]
        lat_s = f"{r['lat_s']:.2f}"
        eng_j = f"{r['eng_j']:.2f}"
        ws    = f"{r['ws_pct']:.1f}" if r["ws_pct"] is not None else "--"
        end   = f"{r['endurance']:.2f}×" if r["endurance"] is not None else "--"
        lines.append(f"{LABELS[m]} & {lat_s} & {eng_j} & {ws} & {end} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    os.makedirs(tex_outdir, exist_ok=True)
    out = os.path.join(tex_outdir, "table_efficiency.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓  {out}")


def table_stt_ablation(sweep_paths, tex_outdir):
    rows = []
    for path in sorted(sweep_paths):
        m = re.search(r"(\d+)", os.path.basename(path))
        if not m: continue
        stt = int(m.group(1))
        d = load_json(path)
        tasks = d["config"].get("tasks", [])
        scores, lats, ws_pcts, endurances = [], [], [], []
        for t in tasks:
            s = get_score(d, t, "tieredkv")
            if s is not None: scores.append(s)
            cs = d.get("results", {}).get(t, {}).get("tieredkv", {}).get("cache_stats", {})
            lat_us = cs.get("modeled_latency_us", 0) or 0
            if lat_us: lats.append(lat_us / 1e6)
            pw = cs.get("paid_writes", 0) or 0
            ws = cs.get("writes_saved", 0) or 0
            dem = cs.get("demoted", 0) or 0
            total = pw + ws
            if total > 0: ws_pcts.append(100.0 * ws / total)
            if pw > 0:    endurances.append(dem / pw)
        rows.append({
            "stt": stt,
            "avg_f1":   round(np.mean(scores), 2) if scores else None,
            "avg_lat_s": round(np.mean(lats), 3)  if lats   else None,
            "ws_pct":   round(np.mean(ws_pcts), 1) if ws_pcts else None,
            "endurance": round(np.mean(endurances), 2) if endurances else None,
        })
    if not rows:
        return

    lines = [
        "\\begin{table}[h]",
        "\\centering\\small",
        "\\caption{TieredKV ablation: STT-RAM budget vs. accuracy and efficiency.}",
        "\\label{tab:stt_ablation}",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "\\textbf{STT Budget} & \\textbf{Avg F1} & \\textbf{Latency (s)} & "
        "\\textbf{Write Sav. (\\%)} & \\textbf{NVM Endur.} \\\\",
        "\\midrule",
    ]
    for r in rows:
        f1  = f"{r['avg_f1']:.2f}" if r["avg_f1"] is not None else "--"
        lat = f"{r['avg_lat_s']:.3f}" if r["avg_lat_s"] is not None else "--"
        ws  = f"{r['ws_pct']:.1f}" if r["ws_pct"] is not None else "--"
        end = f"{r['endurance']:.2f}×" if r["endurance"] is not None else "--"
        lines.append(f"{r['stt']} & {f1} & {lat} & {ws} & {end} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    os.makedirs(tex_outdir, exist_ok=True)
    out = os.path.join(tex_outdir, "table_stt_ablation.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓  {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair",         required=True)
    parser.add_argument("--equal-vram",   required=True, dest="equal_vram")
    parser.add_argument("--sweep-stt",    default=None,  dest="sweep_stt")
    parser.add_argument("--budget-sweep", default=None,  dest="budget_sweep")
    parser.add_argument("--ppl",          default=None)
    parser.add_argument("--latency",      default=None)
    parser.add_argument("--outdir",       default="figures/")
    parser.add_argument("--tex-outdir",   default="results/tables/", dest="tex_outdir")
    args = parser.parse_args()

    setup_style()

    data_fair = load_json(args.fair)
    data_ev   = load_json(args.equal_vram)

    sweep_paths = sorted(glob.glob(args.sweep_stt)) if args.sweep_stt else []

    budget_data_list = []
    if args.budget_sweep:
        pattern_paths = glob.glob(args.budget_sweep)
        # Include checkpoints if explicitly matched or directory passed
        if not pattern_paths and os.path.exists("results/budget_sweep"):
            pattern_paths = glob.glob("results/budget_sweep/budget_*")
        
        seen_budgets = {}
        for path in sorted(pattern_paths):
            m = re.search(r"budget_(\d+)", os.path.basename(path))
            if m:
                b = int(m.group(1))
                # Prefer .json over .ckpt
                if b not in seen_budgets or not path.endswith(".ckpt"):
                    try:
                        seen_budgets[b] = (b, load_json(path))
                    except Exception:
                        pass
        budget_data_list = sorted(list(seen_budgets.values()), key=lambda x: x[0])

    ppl_data = load_json(args.ppl)     if args.ppl     and os.path.exists(args.ppl)     else None
    lat_data = load_json(args.latency) if args.latency and os.path.exists(args.latency) else None

    print(f"\nGenerating publication figures -> {args.outdir}\n")

    fig_accuracy_equivram(data_ev,  args.outdir)
    fig_accuracy_fair(data_fair,    args.outdir)
    fig_f1_vs_budget(budget_data_list, data_fair, args.outdir)
    fig_ppl_vs_context(ppl_data,   args.outdir)
    fig_latency_wallclock(lat_data, args.outdir)
    fig_memory_footprint(lat_data,  args.outdir)
    fig_modeled_latency(data_fair,  args.outdir)
    fig_stt_ablation(sweep_paths,   args.outdir)
    fig_migration_profile(data_ev,  args.outdir)
    fig_nvm_endurance(data_ev, data_fair, args.outdir)
    fig_pareto_scatter(data_fair,   args.outdir)

    print(f"\nGenerating LaTeX tables -> {args.tex_outdir}\n")
    table_accuracy(data_ev,   "Equal-VRAM setting",  args.tex_outdir, "accuracy_equivram")
    table_accuracy(data_fair, "Iso-VRAM setting",    args.tex_outdir, "accuracy_fair")
    table_efficiency(data_fair, args.tex_outdir)
    table_stt_ablation(sweep_paths, args.tex_outdir)


if __name__ == "__main__":
    main()
