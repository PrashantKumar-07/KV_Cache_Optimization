"""Render the source-checked model flow in the original colored visual style.

Run with the repository venv Python; requires only installed matplotlib.
Original JPGs are references and are never modified. SVG text remains editable.
"""
from pathlib import Path
import datetime
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STEM = "tieredkv_flow_diagram_validated"
plt.rcParams.update({"font.family": "Liberation Sans Narrow", "svg.fonttype": "none",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "tieredkv-model-flow-v1"})
fig = plt.figure(figsize=(8, 43 / 3), dpi=96, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 768), ylim=(1376, 0)); ax.axis("off")
ax.add_patch(Rectangle((0,0),768,62,facecolor="#f4f4f6",edgecolor="none"))
labels = []
checks = []


def box(x, y, w, h, shade="0.95", radius=13, width=1.15):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=shade, edgecolor="black", linewidth=width)
    ax.add_patch(patch)
    return (x, y, w, h)


def text(label, x, y, size=15, bold=False, align="left", cite=(), within=None):
    artist = ax.text(x, y, label, fontsize=size, fontweight="bold" if bold else "normal",
        ha=align, va="center", color="black", linespacing=1.15)
    labels.append({"text": label, "source_citations": list(cite),
        "anchor": [x, y], "font_points": size, "bold": bold})
    if within is not None:
        checks.append((artist, within))
    return artist


def arrow(points):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=1.1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="->",
        mutation_scale=12, linewidth=1.1, color="black", shrinkA=0, shrinkB=0))


def row(label, x, y, area, cite=()):
    return text(label, x, y, size=15, cite=cite, within=area)


text("TieredKV: Two-Tier KV Cache", 384, 27, 27, True, "center",
     ["src/tiered_kv_cache.py:187"])
ax.plot([8, 760], [62, 62], color="0.65", linewidth=0.8)
text("Phase 0: Initialization (Once at Prefill)", 384, 87, 22, True, "center")
prompt = box(256, 116, 256, 39, shade="#b7d9f3", radius=19)
text("Prompt Tokens (prefill)", 384, 135, 19, align="center", within=prompt)
arrow([(384, 155), (384, 181)])
initial = box(133, 181, 502, 72)
text("Initial Bifurcation — Observation-Window Scores", 384, 202, 18,
     align="center", cite=["src/tiered_kv_cache.py:213"], within=initial)
text("Actual-query option; key proxy by default", 384, 230, 17,
     align="center", cite=["experiments/longbench_eval.py:705", "experiments/longbench_eval.py:709"], within=initial)
arrow([(133, 217), (128, 217), (128, 279)])
arrow([(384, 253), (384, 279)])
arrow([(635, 217), (640, 217), (640, 279)])
cards = [box(x, 279, 238, 139, shade=color) for x,color in
         zip((9,265,521),("#ffe4c2","#c2e8df","#f2c4c3"))]
card_patches=list(ax.patches)[-3:]
for area,patch,color in zip(cards,card_patches,("#ffca7d","#70c5b9","#df918e")):
    x,y,w,h=area
    band=Rectangle((x,y),w,43,facecolor=color,edgecolor="none")
    band.set_clip_path(patch);ax.add_patch(band)
    ax.plot([x,x+w],[y+43,y+43],color="0.45",linewidth=.7)
for center, heading, lines, citations, area in [
    (128, "Tier 1: GPU VRAM", ["Hot working set", "Sinks + recent + top scores", "Configurable hot budget"],
     ["src/tiered_kv_cache.py:238", "src/tiered_kv_cache.py:253"], cards[0]),
    (384, "Tier 2: Cold KV", ["Simulated STT-RAM tier", "Live rows + later shadows", "Configurable cold budget"],
     ["src/tiered_kv_cache.py:265", "src/tiered_kv_cache.py:606"], cards[1]),
    (640, "Dropped Tokens", ["Not a storage tier", "Beyond both budgets", "Permanently discarded"],
     ["src/tiered_kv_cache.py:246", "src/tiered_kv_cache.py:889"], cards[2]),
]:
    text(heading, center, 300, 18, True, "center", citations, area)
    for y, label in zip((332, 362, 393), lines):
        text(label, center, y, 14.6, align="center", cite=citations, within=area)
ax.plot([8, 760], [439, 439], color="0.45", linewidth=1.1)
text("Per Decode Step (after the first prompt prediction)", 405, 468,
     20, True, "center", ["experiments/longbench_eval.py:1225"])
box(52, 501, 694, 770, shade="0.94", radius=18)
p1 = box(69, 505, 660, 157, shade="#c8dff0", width=0)
p2 = box(69, 675, 660, 135, shade="#d6e7ce", width=0)
p3 = box(69, 823, 660, 133, shade="#fce2c7", width=0)
p4 = box(69, 969, 660, 285, shade="#efc8c7", width=0)

text("1. Prepare Visible KV", 78, 522, 21, True, within=p1,
     cite=["experiments/longbench_eval.py:736"])
for y, label in zip((553, 580, 607, 634), [
    "• Hot rows + quota-limited live cold rows; shadows excluded",
    "• Recency selects cold exposure; same count across layers",
    "• Native option: gather selected cold KV from pinned CPU memory",
    "• Use this prepared set in the next model forward",
]):
    row(label, 84, y, p1, ["experiments/longbench_eval.py:885", "experiments/cold_boundary.py:74", "experiments/cold_boundary.py:81"])
text("2. Model Attention & Feedback", 78, 697, 21, True, within=p2)
for y, label in zip((729, 757, 785), [
    "• Attend hot + selected live cold + current token",
    "• Produce next-token logits and real attention weights",
    "• Update hot scores and cold access timestamps",
]):
    row(label, 84, y, p2, ["experiments/longbench_eval.py:1255", "experiments/longbench_eval.py:1038"])
text("3. Promote Attention-Ranked Cold Rows", 78, 845, 21, True, within=p3)
for y, label in zip((876, 903, 930), [
    "• Apply promotion budget, re-entry gate and live-cold floor",
    "• Inclusive mode: keep a cold shadow backup",
    "• Exclusive ablation: remove the promoted cold copy",
]):
    row(label, 84, y, p3, ["experiments/longbench_eval.py:854", "src/tiered_kv_cache.py:550", "src/tiered_kv_cache.py:606"])
text("4. Evict & Demote", 78, 990, 21, True, within=p4)
row("a) Admit current-token KV; protect sinks and recent tokens", 88, 1020, p4,
    ["src/tiered_kv_cache.py:732", "src/tiered_kv_cache.py:647"])
row("b) Hot overflow: demote the lowest age-normalized score", 88, 1046, p4,
    ["src/tiered_kv_cache.py:687"])
row("    Optional write-aware weight favors rows with backups", 88, 1072, p4,
    ["src/tiered_kv_cache.py:689"])
diamond = Polygon([(390, 1088), (515, 1112), (390, 1136), (265, 1112)],
    closed=True, facecolor="0.96", edgecolor="black", linewidth=1.1)
ax.add_patch(diamond)
text("Shadow backup?", 390, 1112, 14, align="center",
     cite=["src/tiered_kv_cache.py:779"])
arrow([(265, 1112), (218, 1112), (218, 1146)])
arrow([(515, 1112), (596, 1112), (596, 1146)])
text("YES", 238, 1101, 12, align="center")
text("NO", 555, 1101, 12, align="center")
left = box(111, 1146, 214, 47, shade="#f9eceb")
right = box(489, 1146, 214, 47, shade="#f9eceb")
text("Reactivate backup;\nno KV payload write", 218, 1170, 16, align="center",
     cite=["src/tiered_kv_cache.py:790"], within=left)
text("Pay KV write;\nappend live cold row", 596, 1170, 16, align="center",
     cite=["src/tiered_kv_cache.py:809"], within=right)
row("c) Cold overflow: reclaim shadows, then LRU live rows", 88, 1213, p4,
    ["src/tiered_kv_cache.py:880", "src/tiered_kv_cache.py:893"])
row("    Drop permanently; preserve the live floor where feasible", 88, 1238, p4,
    ["src/tiered_kv_cache.py:903"])
arrow([(400, 1271), (400, 1292)])
output = box(304, 1292, 192, 39, shade="#c4e5bd", radius=19)
text("Emit Next Token", 400, 1311, 19, align="center", within=output,
     cite=["experiments/longbench_eval.py:1268"])
arrow([(304, 1311), (18, 1311), (18, 472), (105, 472)])
text("Shadows: reclaim the backup whose hot twin has the highest cumulative score.",
     384, 1339, 10.5, align="center", cite=["src/tiered_kv_cache.py:880"])
text("Cold storage: GPU by default; pinned CPU option. STT costs are analytical.",
     384, 1354, 10.5, align="center",
     cite=["experiments/longbench_eval.py:700", "experiments/cold_boundary.py:81", "src/cost_model.py:1"])
text("Standalone sketch/peek simulation is a separate path; it is not used in this loop.",
     384, 1369, 10.5, align="center",
     cite=["src/cost_model.py:1", "src/tiered_kv_cache.py:945"])

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
violations = []
for artist, (x, y, w, h) in checks:
    bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    xs, ys = sorted([bounds.x0, bounds.x1]), sorted([bounds.y0, bounds.y1])
    if xs[0] < x + 2 or xs[1] > x + w - 2 or ys[0] < y + 2 or ys[1] > y + h - 2:
        violations.append({"label": artist.get_text(), "bounds": [*xs, *ys], "container": [x, y, w, h]})
if violations:
    raise ValueError(json.dumps(violations, indent=2))

for suffix in ("svg", "pdf", "png"):
    metadata = {"Date": None} if suffix == "svg" else (
        {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {})
    fig.savefig(OUT / f"{STEM}.{suffix}", dpi=192, metadata=metadata)
sources = sorted({citation.rsplit(":", 1)[0] for item in labels for citation in item["source_citations"]})
originals = ["figures/tieredkv_flow_diagram.jpg", "figures/tieredkv_flow_diagram_academic.jpg"]
provenance = {
    "scope": "Executed two-retained-tier model decode, ordinary and semantics-preserving native option",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "style_reference": originals[0], "style": "Orange/teal/red initialization cards; blue/green/orange/red decode panels; rounded boxes, condensed sans typography, top-down four-panel loop and backup decision",
    "generator": str(Path(__file__).relative_to(ROOT)), "matplotlib": matplotlib.__version__,
    "canvas_reference_pixels": [768, 1376], "png_pixels": [1536, 2752],
    "editable_vector_text": True, "text_container_checks": len(checks), "text_container_violations": violations,
    "labels": labels, "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    "original_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in originals},
    "outputs_sha256": {f"{STEM}.{s}": hashlib.sha256((OUT / f"{STEM}.{s}").read_bytes()).hexdigest() for s in ("svg", "pdf", "png")},
    "limitations": ["No capacity or bandwidth constants implied", "Native CPU storage is not physical STT", "Actual-query prefill does not imply current-query decode search", "Prepared visibility occurs at end of prior policy call", "Output logits are produced by model before cache bookkeeping", "Grace protection and exclusive ablation are summarized rather than all fallback branches expanded"],
}
(OUT / "color_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"outputs": list(provenance["outputs_sha256"]), "checked_labels": len(checks)}))
