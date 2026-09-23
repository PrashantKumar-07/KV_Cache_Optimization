#!/usr/bin/env python3
"""plot_frac_ablation.py — bifurcation-headroom trade-off figure + table.

Reads the fixed-sample E2 sweep (results/e2_frac*_short.json, TieredKV-only,
same revision/seed/samples): avg F1 vs write-savings% across
sttram_bifurcation_frac = 1.0 / 0.75 / 0.5 / 0.25.

Outputs figures_final/fig8_frac_ablation.pdf/.png + results/tables/table_frac_ablation.tex
Style matches plot_publication_figures.py (NeurIPS/ICML conventions).
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.5,
    "grid.color": "#999999",
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SHORT_TASKS = ["multifieldqa_en", "hotpotqa", "triviaqa", "qasper", "narrativeqa"]


def load_frac(path):
    with open(path) as f:
        d = json.load(f)
    m = re.search(r"frac([0-9_]+)_", os.path.basename(path))
    frac = float(m.group(1).replace("_", "."))
    f1s, savs = [], []
    for t in SHORT_TASKS:
        td = d["results"][t]["tieredkv"]
        cs = td.get("cache_stats", {})
        f1s.append(td["score"])
        tot = cs.get("paid_writes", 0) + cs.get("writes_saved", 0)
        savs.append(100.0 * cs.get("writes_saved", 0) / tot if tot else 0.0)
    n = len(f1s)
    return frac, sum(f1s) / n, sum(savs) / n


def main():
    paths = sorted(glob.glob("results/e2_frac*_short.json"))
    assert len(paths) >= 2, f"need E2 sweep files, found {paths}"
    rows = sorted(load_frac(p) for p in paths)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    fracs = [r[0] for r in rows]
    ax1.plot(fracs, [r[1] for r in rows], "o-", color="#1d3557", label="Avg F1 (5 tasks)")
    ax1.set_xlabel("STT-RAM bifurcation fraction (headroom $\\leftarrow$)")
    ax1.set_ylabel("Avg F1 ($\\times$ 100)", color="#1d3557")
    ax1.tick_params(axis="y", labelcolor="#1d3557")
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.plot(fracs, [r[2] for r in rows], "s--", color="#27ae60", label="Write savings %")
    ax2.set_ylabel("Write savings (%)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_ylim(0, 100)
    ax1.set_title("Headroom Trade-off: Reserving STT-RAM Restores Write Savings\nat Negligible Accuracy Cost (frac=0.75 operating point)",
                  fontsize=10.5, fontweight="bold")
    for f, f1, sv in rows:
        ax1.annotate(f"{f1:.1f}", (f, f1), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    os.makedirs("figures_final", exist_ok=True)
    fig.savefig("figures_final/fig8_frac_ablation.pdf")
    fig.savefig("figures_final/fig8_frac_ablation.png")
    print("saved figures_final/fig8_frac_ablation.pdf/.png")

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/table_frac_ablation.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Bifurcation-fraction sweep (TieredKV, 5 short tasks, "
                "25 samples, fixed revision/seed). frac=0.75 is the operating "
                "point: write savings 7\\% $\\to$ 83\\% at $-$0.16 F1.}\n")
        f.write("\\label{tab:frac_ablation}\n\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("\\textbf{frac} & \\textbf{Avg F1} & \\textbf{Write savings (\\%)} \\\\\n\\midrule\n")
        for fr, f1, sv in rows:
            tag = " \\\\"
            if abs(fr - 0.75) < 1e-9:
                tag = " \\\\ % operating point"
                f.write(f"{fr:.2f} & \\textbf{{{f1:.2f}}} & \\textbf{{{sv:.1f}}}{tag}\n")
            else:
                f.write(f"{fr:.2f} & {f1:.2f} & {sv:.1f}{tag}\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("saved results/tables/table_frac_ablation.tex")
    for fr, f1, sv in rows:
        print(f"  frac={fr:.2f} F1={f1:.2f} sav={sv:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
