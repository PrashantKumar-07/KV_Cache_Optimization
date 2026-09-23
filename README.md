# TieredKV: Tiered KV-Cache Memory Hierarchy for LLM Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Hardware: NVIDIA L40S](https://img.shields.io/badge/Hardware-NVIDIA%20L40S-76b900.svg)](https://www.nvidia.com/en-us/data-center/l40s/)

> **TieredKV** is a research implementation of a two-tier KV-cache memory hierarchy (1024-token VRAM working set + 2048-token STT-RAM victim cache) that exploits STT-RAM's read/write asymmetry to recover the accuracy lost by permanent-eviction baselines — without the full memory cost of keeping every token in VRAM. Dropped tokens are discarded, not offloaded.

---

## Key Idea

Standard KV cache compression methods (H₂O, SnapKV, StreamingLLM) permanently evict tokens from VRAM when the budget fills: a token scored cold at prefill can never come back even if the evolving query needs it later, losing accuracy on long contexts. TieredKV instead **demotes** evicted tokens to a fast STT-RAM tier and **promotes (resurrects)** them back on demand:

```
┌─────────────────────────────────────────────────┐
│   Tier 1 (VRAM)   →   Tier 2 (STT-RAM)          │
│  Hot working set      Warm victim cache         │
│    (GPU fast)         (read-fast/write-slow)    │
│   1024 tokens,          2048 tokens,             │
│   128 MiB bf16          256 MiB bf16             │
└─────────────────────────────────────────────────┘
```
Anything past both budgets is dropped for good — same as the baselines.
Capacities: Mistral-7B GQA holds 32 layers × 8 KV heads × 128 dim × 2 (K,V) × 2 B (bf16) = **128 KiB/token**, so VRAM = 1024 × 128 KiB ≈ 128 MB and STT-RAM = 2048 × 128 KiB ≈ 256 MB (see Fig 11). `sttram_bifurcation_frac` sets STT-RAM's prefill fill level (headline: 0.75 → 1536/2048 resident, 512 rows headroom).

**Inclusive shadow cache:** When a token is promoted from STT-RAM → VRAM, the STT-RAM copy is kept as an immutable shadow backup. If the same token is later re-demoted, the backup is reactivated at **zero write cost**. Because KV tensors never change after prefill, the backup is always valid *whenever it hasn't been reclaimed first*.

That "whenever" matters: this only pays off if STT-RAM has spare capacity to hold backups. If `initial_bifurcation` fills STT-RAM to capacity with cold prompt tokens (the default, `sttram_bifurcation_frac=1.0` — the common case for any document longer than the VRAM+STT budget), every backup gets reclaimed under pressure before it can ever be reused, and write savings measure 0%, independent of eviction-scoring or promotion rate. Reserving headroom (`sttram_bifurcation_frac = 0.75`) restores write savings from 7.2% to 82.9% at −0.17 avg F1 (fixed-sample sweep `results/e2_frac*_short.json`, `figures_final/fig8_frac_ablation.pdf`; cost concentrates on multifieldqa). An earlier version of this README claimed headroom *improved* accuracy; the clean sweep shows a small cost instead, and the old numbers underneath it (mixed 25/15-sample runs across revisions, since archived to `results/archive/`) are withdrawn. See `src/tiered_kv_cache.py`'s `TieredConfig.sttram_bifurcation_frac` docstring for implementation detail.

---

## Architecture

Each decode step runs four phases:

| Phase | Operation |
|-------|-----------|
| **1. Score** | Rank live STT-RAM tokens by the attention the model *actually gave them* on the previous step (`update_scores`), then gate re-entry of tokens that were already VRAM-resident behind a repeated-candidacy streak |
| **2. Promote** | Top-scored STT-RAM tokens migrate → VRAM so future reads hit the fast tier |
| **3. Attention** | Exact attention over the VRAM working set **plus** up to `stt_expose_quota` live STT-RAM tokens, read at STT-RAM's own cost-modelled bandwidth |
| **4. Evict/Demote** | Lowest attention-per-step VRAM tokens demote → STT-RAM (shadow backup reused if available) |

Attention sinks (positions 0–3) and the recent sliding window are pinned and never evicted.

> **On the Quest sketch.** `TieredKVCache.sketch_check()` implements Quest-style
> page-level min/max scoring and is exercised by the CPU simulator and its unit
> tests, but the LongBench harness does **not** call it: once the slow tier
> became attendable, promotion could be driven by the model's real measured
> attention instead of a K-as-Q upper bound, which is strictly better
> information. The sketch remains in the codebase for the simulator and as the
> basis for a future real Quest baseline.

---

## Techniques Integrated

| Technique | Paper | Role in TieredKV |
|-----------|-------|------------------|
| **SnapKV** | Li et al., NeurIPS 2024 | Prompt bifurcation scoring to split tokens at prefill |
| **StreamingLLM** | Xiao et al., ICLR 2024 | Sink + sliding window pinning |
| **Quest** | Tang et al., ICML 2024 | Page-level sketch scoring for STT-RAM candidates |
| **H₂O** | Zhang et al., NeurIPS 2023 | Cumulative-attention eviction policy |
| **FlexGen** | Sheng et al., ICML 2023 | Hierarchical offload (GPU→CPU→disk) — the alternative design point: exact recall via movement instead of our drop-based victim cache |

---

## Results

> ### Current numbers (2026-09-21, reproducible)
>
> All tables below are generated from `results/final_equal_vram_frac075.json`
> (TieredKV at headline `frac=0.75`; baselines carried verbatim from
> `results/final_equal_vram.json`, which they don't depend on — no baseline
> policy reads `sttram_bifurcation_frac`) — one sample count (25/task,
> 10 gov_report), fixed seed 0, `max_ctx=31500` (reference protocol), bf16
> TieredKV residency. The 0.75 TieredKV entries come from the `e2_frac0_75`
> + `frac075_gov` batches, proven bit-identical (scores, counters, modeled
> columns) to final-code batches at matched config — see the file's
> `provenance` block. The 2026-09-07 headline (+5.26 over SnapKV) is
> withdrawn: it ran at 8k truncation, fp32 residency, and unbounded STT
> exposure on a prior revision. See [`AUDIT_VERIFICATION.md`](AUDIT_VERIFICATION.md)
> and git history for the full audit trail.
>
> Measured uncertainty (2026-09-16, `results/tables/ci_equal_vram.json`): per-task
> SEs at n=25 run ±2.5–7.4 F1, so every TieredKV-vs-SnapKV per-task CI overlaps —
> no single-task difference is significant, including our wins. The +0.72 average
> rests on 4/6 task-level wins pointing the same way, not on any one result.
> ≥50 samples/task is needed before the headline margin can be called significant.
>
> Three fairness definitions are reported side by side because the conclusion
> depends on the denominator — a reviewer will construct all three, so we do:
> **equal-VRAM** (TieredKV 1024 VRAM + 2048 STT vs 1024-token baselines),
> **equal-attended** (992 + 32 exposed = 1024 attended), **equal-total**
> (341 + 683 = 1024 resident). No confidence intervals yet: per-sample scores
> are now stored (`sample_scores` in every entry) so SEs are computable, but
> the run-to-run spread on fixed configs (up to ~7 F1 on Qasper at 25 samples
> in older runs) still exceeds the smaller margins below. Treat +0.7 as
> suggestive and +0.05 as a tie.

### Headline: three fair comparisons — real Mistral-7B-Instruct, real LongBench, 2026-09-15

TieredKV attends over its VRAM tier *and* up to `stt_expose_quota` live STT-RAM tokens every step (default quota 32), read at STT-RAM's own cost-modeled bandwidth (STT 50/12.5 GB/s nominal tentpole, VRAM 864 GB/s — see `src/cost_model.py` for cited anchors).

All 6 LongBench tasks, 25 samples/task (10 gov_report), budget=1024. See `figures_final/fig1_accuracy_equivram.pdf`, `fig7_accuracy_fair.pdf`.

| Task | Full-KV | H2O@1024 | SnapKV@1024 | **TieredKV eq-VRAM** | **eq-attended** | **eq-total** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| MultiFieldQA | 60.95 | 56.37 | 54.61 | **55.25** | **55.26** | 42.60 |
| HotpotQA | 48.67 | 39.27 | 37.32 | **37.93** | 33.26 | 27.86 |
| TriviaQA | 74.64 | 64.20 | 64.24 | **67.44** | **67.89** | 64.90 |
| Qasper | 29.55 | 11.47 | 11.54 | **13.28** | **12.37** | 11.11 |
| NarrativeQA | 17.30 | 15.40 | 18.98 | 18.63 | 16.79 | 17.78 |
| GovReport | 32.23 | 27.07 | 28.49 | 26.93 | 27.22 | 22.51 |
| **Average** | 43.89 | 35.63 | 35.86 | **36.58 (+0.72)** | **35.46 (−0.40)** | **31.13 (−4.73)** |

Reading (eq-VRAM column @ headline `frac=0.75`): TieredKV beats SnapKV by +0.72 on average (wins multifieldqa/hotpotqa/triviaqa/qasper, loses narrativeqa/gov_report — note multifieldqa flips to H₂O vs the old `frac=1.0` column, the price of dropping the 2560–3072 tail). At matched attention (992+32, fresh GPU rerun @ `frac=0.75`) it trails by −0.40 — inside the ±2.2 task SEs, i.e. a statistical tie, with the deficit concentrated in hotpotqa (−4.06: the 32-token VRAM cut bites hardest there). It loses clearly at equal total memory (−4.73). The hierarchy wins iff the slow tier adds resident tokens; at fixed total memory the victim-cache overhead (promotion churn, quota-capped visibility) costs more than it recovers. The eq-total column still shows a pre-switch (`frac=1.0`) TieredKV run; the old +5.26 was an attended-tokens artifact (3072 vs 1024) composited with 8k truncation, which lifts baselines less than it lifts full-context methods — at the reference 31.5k context every method scores higher and the gap structure above is what remains.

```latex
\begin{tabular}{lccccc}
\toprule
Task & Full Cache & StreamingLLM & H$_2$O & SnapKV & \textbf{TieredKV (Ours)} \\
\midrule
MultiFieldQA & \textbf{61.0} & 48.4 & 56.4 & 54.6 & 55.2 \\
HotpotQA & \textbf{48.7} & 34.4 & 39.3 & 37.3 & 37.9 \\
TriviaQA & \textbf{74.6} & 64.6 & 64.2 & 64.2 & 67.4 \\
Qasper & \textbf{29.6} & 12.8 & 11.5 & 11.5 & 13.3 \\
NarrativeQA & 17.3 & 17.6 & 15.4 & \textbf{19.0} & 18.6 \\
GovReport & \textbf{32.2} & 26.6 & 27.1 & 28.5 & 26.9 \\
\midrule
\textbf{Average} & \textbf{43.9} & 34.1 & 35.6 & 35.9 & 36.6 \\
\bottomrule
\end{tabular}
```
(equal-VRAM column; `results/tables/table1_accuracy_equivram.tex` is the generated source of truth.)

**Latency (honest ledger).** Remeasured 2026-09-16 at the headline config with promotion live (`results/latency_headline.json`, synthetic prompt so timing-only): ITL is flat across context — TieredKV 25.1 ms/tok at 16k ctx vs H2O/SnapKV ~24.8 (~+1.5%) and vs Full-KV 61.7 (2.5× faster, since full attention grows O(n)). The old ~3–5.5× gap dated to pre-tensor bookkeeping and unbounded exposure; wall-clock on real LongBench text post-fix is still unmeasured and may differ (non-uniform attention drives more promotion churn than a repeated prompt). Modeled cost at the cited nominal STT tentpole is tracked separately (`results/tables/table3_efficiency.tex`).

**STT tentpole sensitivity (2026-09-21, analytical replay from saved counters — no new GPU needed).** Varying STT bandwidth over the three cited anchors in `src/cost_model.py` (pessimistic 21/5.25, nominal 50/12.5, optimistic 100/25 GB/s; Everspin datasheet / Li et al. 2024 / Melody ASPLOS'25 / CXL-Bench 2025) moves only TieredKV's bill — baselines never touch STT. TieredKV migration latency replays exactly from saved event counters: **0.803 / 0.346 / 0.180 s** (pess/nom/opt). Total modeled energy is tentpole-invariant (**70.63 J** — pJ/bit is fixed by methodology) and stays below H₂O (72.01 J) at all three anchors; write savings are event ratios, likewise unaffected. The qualitative position (modeled latency punishes TieredKV's extra traffic while wall-clock Fig 4 shows parity) holds at every anchor. Limitation recorded honestly: per-step records (peeked/sketch/exposed-step distributions) aren't exported to the JSONs, so only the migration slice — not the full tower — replays exactly; future runs should export Σpeeked/Σexposed-steps/Σsketch-bytes (one-line `stats()` change).

**What changed to get here** — `TieredKVPolicy._visible_tuple()` builds the attended set from VRAM + quota-capped live STT-RAM (uniform length across layers for HF's shared causal mask). Promotion is driven by the model's real measured attention on STT-RAM tokens (not a K-as-Q sketch proxy), gated by `resolve_promotions()` hysteresis. A `stt_live_floor` (default 1) keeps one live STT row per layer on both the promote and evict paths so one drained layer can't veto exposure for all 32; unanimous-empty tiers (short docs) count as `stt_unused_steps`, not starvation. An attempt to inject *peeked* STT-RAM content directly into real attention was tried and rolled back: HF shares one causal mask across all 32 layers (rolled back after measuring ~20x promotion collapse). Prior harness bugs fixed along the way: cross-sample stats are accumulated (not last-sample-only), dropped tokens bill `drop_cost` (not a phantom DRAM write), STT-RAM figures are cited tentpoles (not hand-picked), checkpoints are fingerprinted including sample counts.

### STT-RAM budget ablation (clean)

VRAM fixed at 1024, STT-RAM swept 512→3072, 5 short tasks × 25 samples, bf16, final code (`results/clean_sweep_stt/`; full numbers in `results/tables/table4_stt_ablation.tex` — view with `experiments/preview_tables.py`).

| STT-RAM budget | 512 | 1024 | 2048 | 3072 |
|---|:---:|:---:|:---:|:---:|
| Avg F1 | 38.19 | 38.06 | 38.67 | 38.27 |

Flat within noise (±0.6) — the old monotonic 35.77→44.98 curve does not survive clean measurement (3 tasks × 10 samples across mixed revisions). With VRAM fixed and exposure quota-bounded, extra slow-tier capacity beyond working-set needs adds retrievable pool the fixed window cannot surface; per-task directions split (triviaqa falls 70.1→66.6 with bigger STT, hotpotqa rises 37.4→38.9). Reported as measured: capacity without visibility. Window-size ablation (`experiments/sweep_window_size.sh`) is scripted but not yet run.

### Bifurcation-headroom operating point (new)

Reserving 25% STT headroom (`sttram_bifurcation_frac=0.75`, fixed-sample sweep `results/e2_frac*_short.json`, `figures_final/fig8_frac_ablation.pdf`): write savings 7.2% → **82.9%** at −0.17 avg F1 (38.67→38.51, noise; cost concentrates on multifieldqa −1.9, others flat-or-better). Default `frac=1.0` pre-fills STT at prefill and structurally guarantees ~0% savings — the old 0% was a capacity artifact, proven by the infinite-STT oracle (`results/e1_oracle_short.json`: 77–88% savings, F1 flat). This is the paper's write-savings claim, with its trade-off stated.

---

## Repository Structure

```
Project/                          ← shared root (sibling of multimodal_document_rag/)
├── kv_cache/                     ← THIS project (run all commands from here)
│   ├── src/
│   │   ├── attention.py            # Shared SDPA math + MAC counting
│   │   ├── tiered_kv_cache.py      # Core 2-tier cache (TieredConfig, TieredKVCache)
│   │   ├── baselines.py            # FullAttention, StreamingLLM, H₂O, SnapKV, Quest
│   │   ├── cost_model.py           # Analytical latency/energy model (L40S-calibrated)
│   │   └── metrics.py              # Per-step and aggregate metrics
│   ├── experiments/
│   │   ├── longbench_eval.py       # ★ Main: LongBench SOTA comparison, all policies
│   │   ├── longbench_metrics.py    # F1, ROUGE-L, substring-match scorers
│   │   └── plot_sota_comparison.py # Bar chart + radar chart from LongBench JSON
│   ├── sota/
│   │   ├── snapkv/                 # FasterDecoding/SnapKV (upstream)
│   │   ├── streamingllm/           # mit-han-lab/streaming-llm (upstream)
│   │   ├── h2o/                    # FMInference/H2O (upstream)
│   │   └── quest/                  # mit-han-lab/Quest (upstream)
│   ├── figures/                    # Generated publication figures (PNG/JPG)
│   ├── results/                    # JSON result files (gitignored from build outputs)
│   ├── tests/
│   │   ├── test_tiered_cache.py    # 2-tier cache invariants
│   │   ├── test_sketch_bound.py    # Quest bound property tests
│   │   ├── test_baselines.py       # baseline budget/sink invariants
│   │   ├── test_cost_model.py      # latency/energy unit conversions
│   │   └── test_longbench_policies.py  # harness integration tests
│   ├── TieredKV_Project_Explanation.md   # Full project write-up
│   ├── TieredKV_Project_Explanation.pdf  # PDF version with embedded figures
│   └── requirements.txt
├── venv_kvcache/                 ← this project's venv (prebuilt — use it)
├── models/                       ← local model weights (mistral-7b-instruct)
└── hf_cache/                     ← Hugging Face download cache
```

---

## Environment Setup

**Requirements:** Python 3.10+, CUDA GPU (tested on NVIDIA L40S, CUDA 12.4, 48 GB VRAM).

> ⚠️ Use **this project's venv** (`Project/venv_kvcache`) — not `multimodal_document_rag/.venv`, which lacks torch/datasets and will fail with `ModuleNotFoundError`. All commands below assume your shell is in `Project/kv_cache/`.

```bash
# 1. Activate the prebuilt environment (from Project/kv_cache/)
source ../venv_kvcache/bin/activate

# ---- Only if you need to recreate it from scratch: ----
# /usr/bin/python3 -m venv ../venv_kvcache
# ../venv_kvcache/bin/pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# ../venv_kvcache/bin/pip install -r requirements.txt
# (adjust cu124 to your CUDA version)

# 2. Set Hugging Face cache directory (must have 20+ GB free)
export HF_HOME=/path/to/your/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Download Mistral-7B-Instruct-v0.2

```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir ../models/mistral-7b-instruct \
    --max-workers 8
```

> For Llama-3-8B: accept the license at [huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), run `huggingface-cli login`, then replace the model path.

---

## Quick Verification (No Model Required)

```bash
# Unit + integration tests (50 tests, CPU + tiny-model harness).
# Run from Project/kv_cache/ with Project/venv_kvcache active:
python -m pytest tests/ -v
```

---

## Reproducing All Results

### Step 1 — Baselines at budget=1024 (Full, StreamingLLM, H2O, SnapKV)

```bash
MODEL=../models/mistral-7b-instruct

python experiments/longbench_eval.py --model $MODEL \
    --tasks multifieldqa_en qasper narrativeqa hotpotqa gov_report triviaqa \
    --methods full streamingllm h2o snapkv \
    --budget 1024 --device cuda \
    --json results/baselines.json
```

### Step 2 — TieredKV at equal VRAM (the headline comparison)

Same VRAM footprint as the baselines above (1024 tokens fast-tier), plus a slow tier on top:

```bash
python experiments/longbench_eval.py --model $MODEL \
    --tasks multifieldqa_en qasper narrativeqa hotpotqa gov_report triviaqa \
    --methods tieredkv \
    --budget 1024 --tiered-vram-budget 1024 --tiered-stt-budget 2048 \
    --device cuda \
    --json results/tieredkv_equal_vram.json
```

> **Note:** Full evaluation takes several hours (TieredKV currently has a known, unfixed latency overhead — see Results). Use `--max-samples 20` for a quick preview.

### Step 3 — Comparison Figures

```bash
python experiments/plot_publication_figures.py \
    --equal-vram results/final_equal_vram_frac075.json \
    --fair results/final_equal_attended_frac075.json \
    --sweep-glob "results/clean_sweep_stt/stt*.json" \
    --outdir figures_final/
python experiments/plot_frac_ablation.py
```

Output: `figures_final/fig1_accuracy_equivram.pdf/.png`, `fig7_accuracy_fair.pdf/.png` (merge Step 1/2's JSON files into one `config`/`results` file first — see `results/final_equal_vram.json` for the schema, including its `provenance` block).

---

## Hardware & Cost Model

Calibrated analytically for **NVIDIA L40S** (tested hardware):

| Tier | Technology | Read BW | Write BW | Energy |
|------|-----------|---------|---------|--------|
| Tier 1 — VRAM | GDDR6 | 864 GB/s | 864 GB/s | 12 pJ/bit |
| Tier 2 — STT-RAM | STT-MRAM (simulated) | 50 GB/s | 12.5 GB/s | 2 / 8 pJ/bit |

The 4× write/read asymmetry in STT-RAM is exactly what the inclusive victim cache exploits — by reusing shadow backups, it avoids the slow/expensive write path entirely.

---

## Running Tests

```bash
# From Project/kv_cache/ with Project/venv_kvcache activated:
source ../venv_kvcache/bin/activate
python -m pytest tests/ -v
```

All 50 tests cover: bifurcation and capacity invariants, the promote/demote cycle, inclusive write savings, the peek/promote hysteresis gate, the Quest page bound's upper-bound property, baseline budget accounting and sink survival, cost-model unit conversions (including the slow-tier-slower-than-fast-tier guard), live-floor starvation guards on both promote and evict paths, unused-vs-starved accounting, write-aware demotion ordering, reset() single-counting, drop-cost billing, and `longbench_eval.py` integration (per-policy budget limits, visible-tuple length uniformity, bounded slow-tier exposure, true RoPE positions, and cross-sample stats accumulation) against a tiny randomly-initialised Mistral.

---

## Documentation

Full project documentation including architecture, evaluation metrics, benchmark analysis, and figure explanations is available in:

- [`TieredKV_Project_Explanation.md`](TieredKV_Project_Explanation.md) — Markdown version
- [`TieredKV_Project_Explanation.pdf`](TieredKV_Project_Explanation.pdf) — PDF with embedded figures

---

## Citation

If you use this code, please cite:

```bibtex
@misc{kumar2025tieredkv,
  title   = {TieredKV: Exploiting STT-RAM Read/Write Asymmetry for
             Near-Lossless KV Cache Compression in LLM Inference},
  author  = {Kumar, Prashant},
  year    = {2025},
  url     = {https://github.com/PrashantKumar-07/KV_Cache_Optimization}
}
```

---

## License

[MIT License](LICENSE) — © 2025 Prashant Kumar
