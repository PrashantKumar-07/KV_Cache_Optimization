"""
plot_results.py  (LOCAL-ONLY visualisation -- NOT part of the server pipeline)
==============================================================================
Turn the numbers the experiments already emit into paper-ready figures.

WHY THIS IS A SEPARATE, OPTIONAL SCRIPT
---------------------------------------
The core code (src/ + the experiment drivers) is deliberately dependency-light
(torch + numpy only) so it runs unchanged on Server 3.  Plotting needs
matplotlib, which we do NOT want to force onto the server.  So this script is
quarantined: it imports matplotlib, reads the JSON/CSV that the experiments
produce, and writes PNGs.  Workflow:

    server:  python experiments/sweep_sram.py --csv results/sweep.csv
             python experiments/model_wrapper.py ... --json results/llama_layers.json
    (copy results/*.json and *.csv back to your laptop)
    laptop:  python experiments/plot_results.py results/sweep.csv results/llama_layers.json
             python experiments/plot_results.py results/tiered_run.json

The three input formats are auto-detected by content, so you can pass any mix:

  1. sweep_sram CSV        -> accuracy-vs-budget, latency-vs-budget,
                              migration-volume-vs-budget, accuracy/latency Pareto
  2. model_wrapper JSON    -> per-layer accuracy, per-layer migration volume
     (has "per_layer" key)    (stacked), tiered-vs-oracle GOPs, per-layer latency
  3. RunMetrics JSON       -> tier occupancy over time, per-step latency
     (has "steps" key)        breakdown (sketch/promote/attention/demote),
                              cumulative migration over time

Every figure is saved as a PNG next to the input file (or under --outdir).
Nothing here is imported by the core code.

Usage:
  python experiments/plot_results.py results/sweep.csv
  python experiments/plot_results.py results/*.json --outdir figures
  python experiments/plot_results.py results/sweep.csv results/llama_layers.json results/tiered_run.json
"""

import os
import sys
import csv
import json
import argparse

import matplotlib
matplotlib.use("Agg")            # headless: no display needed, safe over SSH
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- io
def _slug(path):
    """Base filename (no extension) to prefix the figures this input produces."""
    return os.path.splitext(os.path.basename(path))[0]


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, name + ".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def _is_number(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and x != x)


def detect_format(path):
    """Return one of 'sweep_csv', 'model_wrapper', 'runmetrics', or None."""
    if path.lower().endswith(".csv"):
        return "sweep_csv"
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(obj, dict) and "per_layer" in obj:
        return "model_wrapper"
    if isinstance(obj, dict) and "steps" in obj:
        return "runmetrics"
    return None


# ------------------------------------------------------------------ sweep (CSV)
def load_sweep_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # numeric coercion; keep NaN for blank/"nan" accuracy cells
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    for r in rows:
        for k in ("sram", "stt", "acc_all", "acc_anchor", "gops",
                  "migrate", "our_lat_us", "net_speedup_us"):
            if k in r:
                r[k] = num(r[k])
    rows.sort(key=lambda r: r["sram"])
    return rows


def plot_sweep(path, outdir):
    rows = load_sweep_csv(path)
    if not rows:
        print(f"  (skip) no rows in {path}")
        return
    slug = _slug(path)
    sram = [r["sram"] for r in rows]

    # 1. accuracy vs SRAM budget (all + anchor) -- the "money" curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sram, [r["acc_all"] for r in rows], "o-", label="Acc (all steps)")
    if any(_is_number(r.get("acc_anchor")) for r in rows):
        ax.plot(sram, [r["acc_anchor"] for r in rows], "s--",
                label="Acc (long-range anchor)")
    ax.set_xlabel("SRAM budget (tokens)")
    ax.set_ylabel("cosine accuracy vs full-attention oracle")
    ax.set_title("Accuracy vs SRAM budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir, f"{slug}_accuracy_vs_budget")

    # 2. derived latency vs SRAM budget (our cost + net vs full baseline)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sram, [r["our_lat_us"] for r in rows], "o-", label="TieredKV latency")
    full_lat = [r["our_lat_us"] + r["net_speedup_us"] for r in rows]  # full = ours + net
    ax.plot(sram, full_lat, "^:", label="Full-attention baseline")
    ax.set_xlabel("SRAM budget (tokens)")
    ax.set_ylabel("derived latency (us, total run)")
    ax.set_title("Derived latency vs SRAM budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir, f"{slug}_latency_vs_budget")

    # 3. migration volume vs SRAM budget
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sram, [r["migrate"] for r in rows], "d-", color="tab:red")
    ax.set_xlabel("SRAM budget (tokens)")
    ax.set_ylabel("total migrations (promoted+demoted+dropped)")
    ax.set_title("Migration volume vs SRAM budget")
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, f"{slug}_migration_vs_budget")

    # 4. accuracy / latency Pareto (each point a budget; label with SRAM size)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["our_lat_us"] for r in rows]
    ys = [r["acc_all"] for r in rows]
    ax.plot(xs, ys, "o-", color="tab:purple")
    for r in rows:
        ax.annotate(f"{int(r['sram'])}", (r["our_lat_us"], r["acc_all"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("derived latency (us, total run)")
    ax.set_ylabel("cosine accuracy vs oracle")
    ax.set_title("Accuracy vs latency Pareto (labels = SRAM budget)")
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, f"{slug}_pareto_accuracy_latency")

    # 5. sweet-spot / verdict figure: mark the crossover budget where TieredKV
    #    first beats full-attention (first "WIN" in the verdict column)
    if all("verdict" in r for r in rows):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"WIN": "green", "fast": "orange", "slow": "red"}
        for verdict, color in colors.items():
            subset = [r for r in rows if r.get("verdict") == verdict]
            if subset:
                ax.scatter([r["sram"] for r in subset],
                          [r["acc_all"] for r in subset],
                          c=color, label=verdict, s=60, alpha=0.7)
        # mark the crossover (first WIN)
        wins = [r for r in rows if r.get("verdict") == "WIN"]
        if wins:
            first_win = wins[0]
            ax.axvline(first_win["sram"], color="green", linestyle="--", linewidth=2,
                      label=f"crossover @ {int(first_win['sram'])} tokens")
            ax.scatter([first_win["sram"]], [first_win["acc_all"]],
                      c="green", s=200, marker="*", edgecolors="black", linewidths=1.5,
                      zorder=5)
        ax.set_xlabel("SRAM budget (tokens)")
        ax.set_ylabel("cosine accuracy vs oracle")
        ax.set_title("Sweet-spot: crossover budget (first WIN over full-attention)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, outdir, f"{slug}_sweet_spot_verdict")


# ------------------------------------------------------- model_wrapper per-layer
def plot_model_wrapper(path, outdir):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    per_layer = obj.get("per_layer", [])
    if not per_layer:
        print(f"  (skip) empty per_layer in {path}")
        return
    slug = _slug(path)
    idx = list(range(len(per_layer)))

    # 1. per-layer accuracy (all / anchor / local where present)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(idx, [r["acc_all"] for r in per_layer], "o-", label="Acc (all)")
    if any(_is_number(r.get("acc_anchor")) for r in per_layer):
        ax.plot(idx, [r.get("acc_anchor", float("nan")) for r in per_layer],
                "s--", label="Acc (anchor)")
    if any(_is_number(r.get("acc_local")) for r in per_layer):
        ax.plot(idx, [r.get("acc_local", float("nan")) for r in per_layer],
                "^:", label="Acc (local)")
    ax.set_xlabel("layer index")
    ax.set_ylabel("cosine accuracy vs oracle")
    ax.set_title("Per-layer accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir, f"{slug}_per_layer_accuracy")

    # 2. per-layer migration volume, stacked (promoted/demoted/dropped)
    fig, ax = plt.subplots(figsize=(7, 4))
    prom = [r["promoted"] for r in per_layer]
    demo = [r["demoted"] for r in per_layer]
    drop = [r["dropped"] for r in per_layer]
    ax.bar(idx, prom, label="promoted STT->SRAM")
    ax.bar(idx, demo, bottom=prom, label="demoted SRAM->STT")
    ax.bar(idx, drop, bottom=[p + d for p, d in zip(prom, demo)],
           label="dropped STT->cold")
    ax.set_xlabel("layer index")
    ax.set_ylabel("migration events")
    ax.set_title("Per-layer migration volume (thrashing visible here)")
    ax.legend()
    _save(fig, outdir, f"{slug}_per_layer_migration")

    # 3. tiered vs oracle GOPs per layer (compute saved)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    ax.bar([i - width / 2 for i in idx], [r["tiered_GOPs"] for r in per_layer],
           width, label="TieredKV")
    ax.bar([i + width / 2 for i in idx], [r["oracle_GOPs"] for r in per_layer],
           width, label="Full oracle")
    ax.set_xlabel("layer index")
    ax.set_ylabel("GOPs")
    ax.set_title("Compute: TieredKV vs full attention (per layer)")
    ax.legend()
    _save(fig, outdir, f"{slug}_per_layer_gops")

    # 4. per-layer derived latency
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(idx, [r["latency_us"] for r in per_layer], "o-", color="tab:green")
    ax.set_xlabel("layer index")
    ax.set_ylabel("derived latency (us)")
    ax.set_title("Per-layer derived latency")
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, f"{slug}_per_layer_latency")


# ---------------------------------------------------------- RunMetrics per-step
def plot_runmetrics(path, outdir):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    steps = obj.get("steps", [])
    if not steps:
        print(f"  (skip) empty steps in {path}")
        return
    slug = _slug(path)
    t = [s["step"] for s in steps]

    # 1. tier occupancy over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, [s["sram_tokens"] for s in steps], label="SRAM")
    ax.plot(t, [s["sttram_tokens"] for s in steps], label="STT-RAM")
    if any(s.get("dram_tokens", 0) for s in steps):
        ax.plot(t, [s["dram_tokens"] for s in steps], label="DRAM")
    ax.set_xlabel("decode step")
    ax.set_ylabel("resident tokens")
    ax.set_title("Tier occupancy over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir, f"{slug}_occupancy_over_time")

    # 2. per-step latency breakdown, stacked
    #    (sketch / promote / attention / demote -- the migration-overhead anatomy)
    fig, ax = plt.subplots(figsize=(7, 4))
    sk = [s.get("lat_sketch_us", 0.0) for s in steps]
    pr = [s.get("lat_promote_us", 0.0) for s in steps]
    at = [s.get("lat_attention_us", 0.0) for s in steps]
    de = [s.get("lat_demote_us", 0.0) for s in steps]
    ax.stackplot(t, sk, pr, at, de,
                 labels=["sketch", "promote (STT read)",
                         "attention (SRAM read)", "demote (STT write)"])
    ax.set_xlabel("decode step")
    ax.set_ylabel("derived latency (us)")
    ax.set_title("Per-step latency breakdown (migration-overhead anatomy)")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, outdir, f"{slug}_latency_breakdown")

    # 3. cumulative migration over time
    fig, ax = plt.subplots(figsize=(7, 4))
    def cumsum(key):
        acc, out = 0, []
        for s in steps:
            acc += s.get(key, 0)
            out.append(acc)
        return out
    ax.plot(t, cumsum("promoted_tokens"), label="promoted (cum)")
    ax.plot(t, cumsum("demoted_tokens"), label="demoted (cum)")
    ax.plot(t, cumsum("dropped_tokens"), label="dropped (cum)")
    ax.set_xlabel("decode step")
    ax.set_ylabel("cumulative migrations")
    ax.set_title("Cumulative migration volume over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir, f"{slug}_migration_over_time")


# --------------------------------------------------------------------- dispatch
DISPATCH = {
    "sweep_csv": plot_sweep,
    "model_wrapper": plot_model_wrapper,
    "runmetrics": plot_runmetrics,
}
HUMAN = {
    "sweep_csv": "sweep_sram CSV",
    "model_wrapper": "model_wrapper per-layer JSON",
    "runmetrics": "RunMetrics per-step JSON",
}


def main():
    ap = argparse.ArgumentParser(
        description="Render paper figures from experiment outputs "
                    "(LOCAL-only; needs matplotlib, not used on the server).")
    ap.add_argument("inputs", nargs="+",
                    help="one or more results files (sweep CSV / model_wrapper "
                         "JSON / tiered_run JSON). Format is auto-detected.")
    ap.add_argument("--outdir", default=None,
                    help="directory for PNGs (default: alongside each input).")
    args = ap.parse_args()

    made = 0
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"[skip] not a file: {path}")
            continue
        fmt = detect_format(path)
        if fmt is None:
            print(f"[skip] unrecognised format: {path}")
            continue
        outdir = args.outdir or (os.path.dirname(os.path.abspath(path)) or ".")
        print(f"[{HUMAN[fmt]}] {path}")
        DISPATCH[fmt](path, outdir)
        made += 1

    if made == 0:
        print("\nNo recognised inputs. Produce some first, e.g.:")
        print("  python experiments/sweep_sram.py --csv results/sweep.csv")
        print("  python experiments/model_wrapper.py --smoke --json results/smoke.json")
        print("  python experiments/compare_accuracy.py   # writes results/tiered_run.json")
        sys.exit(1)
    print(f"\nDone: rendered figures for {made} input(s).")


if __name__ == "__main__":
    main()
