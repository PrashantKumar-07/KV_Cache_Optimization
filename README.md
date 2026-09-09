# TieredKV: 3-Tier KV Cache Memory Hierarchy for LLM Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Hardware: NVIDIA L40S](https://img.shields.io/badge/Hardware-NVIDIA%20L40S-76b900.svg)](https://www.nvidia.com/en-us/data-center/l40s/)

> **TieredKV** is a research implementation of a three-tier KV cache memory hierarchy that exploits STT-RAM's read/write asymmetry to recover the accuracy lost by permanent-eviction baselines — without the full memory cost of keeping every token in VRAM.

---

## Key Idea

Standard KV cache compression methods (H₂O, SnapKV, StreamingLLM) permanently evict tokens from VRAM when the budget fills, losing accuracy on long contexts. TieredKV instead **demotes** evicted tokens to a fast STT-RAM tier and **promotes** them back on demand:

```
┌─────────────────────────────────────────────────────────────┐
│   Tier 1 (VRAM)   →   Tier 2 (STT-RAM)   →   Tier 3 (DRAM) │
│  Hot working set      Warm victim cache      Cold offload    │
│    (GPU fast)         (read-fast/write-slow)   (CPU DRAM)    │
└─────────────────────────────────────────────────────────────┘
```

**Inclusive shadow cache:** When a token is promoted from STT-RAM → VRAM, the STT-RAM copy is kept as an immutable shadow backup. If the same token is later re-demoted, the backup is reactivated at **zero write cost**. Because KV tensors never change after prefill, the backup is always valid *whenever it hasn't been reclaimed first*.

That "whenever" matters: this only pays off if STT-RAM has spare capacity to hold backups. If `initial_bifurcation` fills STT-RAM to capacity with cold prompt tokens (the default, `sttram_bifurcation_frac=1.0` — the common case for any document longer than the VRAM+STT budget), every backup gets reclaimed under pressure before it can ever be reused, and write savings measure 0%, independent of eviction-scoring or promotion rate. Reserving headroom (`sttram_bifurcation_frac < 1.0`) restores substantial write savings (measured 74-85% across a real 3-task, 12-samples/task LongBench pilot) and, in that same pilot, *improved* accuracy rather than costing it — see the Results section below for the full sweep and its caveats (small sample size; treat as a promising pilot, not a settled result). See `src/tiered_kv_cache.py`'s `TieredConfig.sttram_bifurcation_frac` docstring for implementation detail.

---

## Architecture

Each decode step runs four phases:

| Phase | Operation |
|-------|-----------|
| **1. Sketch** | Quest-style page-level min/max scoring over STT-RAM pages to identify promotion candidates cheaply |
| **2. Promote** | Top-scored STT-RAM pages migrate → VRAM |
| **3. Attention** | Exact scaled dot-product attention over the VRAM working set only |
| **4. Evict/Demote** | Lowest cumulative-attention VRAM tokens demote → STT-RAM (shadow backup reused if available) |

Attention sinks (positions 0–3) and the recent sliding window are pinned and never evicted.

---

## Techniques Integrated

| Technique | Paper | Role in TieredKV |
|-----------|-------|------------------|
| **SnapKV** | Li et al., NeurIPS 2024 | Prompt bifurcation scoring to split tokens at prefill |
| **StreamingLLM** | Xiao et al., ICLR 2024 | Sink + sliding window pinning |
| **Quest** | Tang et al., ICML 2024 | Page-level sketch scoring for STT-RAM candidates |
| **H₂O** | Zhang et al., NeurIPS 2023 | Cumulative-attention eviction policy |
| **FlexGen** | Sheng et al., ICML 2023 | DRAM offload path (Tier 3) |

---

## Results

### Headline: equal-VRAM comparison — real Mistral-7B-Instruct, real LongBench, 2026-09-07

TieredKV attends over its VRAM tier *and* its live STT-RAM tier every step (not VRAM alone — an earlier version silently gave TieredKV a 3x smaller effective context than every baseline at the same nominal budget, explained below). That makes an equal-VRAM comparison meaningful: TieredKV given the SAME VRAM footprint as H2O/SnapKV (1024 tokens), plus a 2048-token STT-RAM tier read at its own (cost-modeled, non-free) bandwidth.

All 6 original LongBench tasks, 25 samples/task (10 for gov_report), budget=1024. See [`figures/fig_longbench_comparison.png`](figures/fig_longbench_comparison.png), [`figures/fig_radar_comparison.png`](figures/fig_radar_comparison.png).

| Task | Full-KV | H2O@1024 | SnapKV@1024 | **TieredKV (equal-VRAM)** |
|------|:---:|:---:|:---:|:---:|
| MultiFieldQA | 60.95 | 55.69 | 55.81 | **63.57** |
| HotpotQA | 48.67 | 39.66 | 37.32 | **46.64** |
| TriviaQA | 66.08 | 64.00 | 60.92 | **67.22** |
| Qasper | 29.30 | 11.18 | 13.51 | **19.79** |
| NarrativeQA | 17.36 | 15.23 | 17.73 | **17.62** |
| GovReport | 18.63 | 14.78 | 16.61 | **18.63** |
| **Average** | 40.17 | 33.42 | 33.65 | **38.91** |

TieredKV beats both SOTA baselines by ~5.3-5.5 points on average at equal VRAM, is within 1.25 points of Full-KV, exceeds Full-KV outright on 3/6 tasks, and ties it exactly on GovReport. This is a legitimate systems argument, not a trick: a real memory hierarchy attends over more total context than a VRAM-only baseline at the same fast-tier footprint, because the slow tier is read (at a real, cost-modeled bandwidth penalty), not just used as an invisible staging buffer.

```latex
\begin{tabular}{lccccc}
\toprule
Task & Full Cache & StreamingLLM & H$_2$O & SnapKV & \textbf{TieredKV (Ours)} \\
\midrule
MultiFieldQA & 61.0 & 48.4 & 55.7 & 55.8 & \textbf{63.6} \\
HotpotQA & \textbf{48.7} & 34.2 & 39.7 & 37.3 & 46.6 \\
TriviaQA & 66.1 & 64.6 & 64.0 & 60.9 & \textbf{67.2} \\
Qasper & \textbf{29.3} & 12.2 & 11.2 & 13.5 & 19.8 \\
NarrativeQA & 17.4 & 17.2 & 15.2 & \textbf{17.7} & 17.6 \\
GovReport & \textbf{18.6} & 14.4 & 14.8 & 16.6 & \textbf{18.6} \\
\midrule
\textbf{Average} & \textbf{40.2} & 31.8 & 33.4 & 33.6 & 38.9 \\
\bottomrule
\end{tabular}
```

**Known open issue — latency.** Measured directly (real Mistral-7B, gov_report): TieredKV is currently **~3-5.5x slower per token than H2O** on raw GPU wall-clock. Part of this is expected and not a bug: at equal-VRAM, TieredKV attends 3072 tokens vs. H2O's 1024, and this GPU has no real separate slow-memory hardware, so more attended tokens costs proportionally more compute regardless of tier. Two rounds of vectorizing Python-level per-token loops in `evict_and_demote`/overflow handling (`src/tiered_kv_cache.py`) improved but did not close the remaining gap at *equal total* tokens. The paper's actual latency/energy claim should use the **modeled** cost (`TieredKVPolicy.stats()["modeled_latency_us"/"modeled_energy_nj"]`, from `cost_model.py`'s NVM-literature bandwidth figures), which is what a real STT-RAM-backed system would experience — not raw wall-clock on a GPU-only prototype with no physically separate slow tier.

**What changed to get here** — `TieredKVPolicy._visible_tuple()` (`experiments/longbench_eval.py`) now builds what the model attends over from VRAM + live (non-shadow) STT-RAM tokens, not VRAM alone. Promotion is driven by the model's real measured attention on STT-RAM tokens (not a K-as-Q sketch proxy), gated by `TieredKVCache.resolve_promotions()`'s hysteresis so a token that already proved not durably valuable can't instantly re-promote on one new hit. An attempt to also inject *peeked* (not-yet-promoted) STT-RAM content directly into the model's real attention was tried and rolled back: HF shares one causal mask across all 32 decoder layers, and independent per-layer peek decisions broke that invariant, measured to collapse promotion volume ~20x and cost double-digit F1 points before being caught and reverted.

### STT-RAM budget ablation

VRAM fixed at 1024 (equal to every baseline), STT-RAM budget swept 512→3072 (3-task subset, 10 samples/task). See [`figures/fig_ablation_stt_budget.png`](figures/fig_ablation_stt_budget.png).

| STT-RAM budget | 512 | 1024 | 2048 (headline) | 3072 |
|---|:---:|:---:|:---:|:---:|
| Avg F1 | 35.77 | 38.79 | 40.91 | 44.98 |

Accuracy increases monotonically with slow-tier size, no diminishing returns visible yet even at 3x VRAM — direct evidence for the core claim, and confirms the headline 2048 config isn't a cherry-picked peak. Window-size ablation (`experiments/sweep_window_size.sh`) is scripted but not yet run.

---

## Repository Structure

```
KV_Cache_Optimization/
├── src/
│   ├── attention.py            # Shared SDPA math + MAC counting
│   ├── tiered_kv_cache.py      # Core 3-tier cache (TieredConfig, TieredKVCache)
│   ├── baselines.py            # FullAttention, StreamingLLM, H₂O, SnapKV, Quest
│   ├── cost_model.py           # Analytical latency/energy model (L40S-calibrated)
│   └── metrics.py              # Per-step and aggregate metrics
├── experiments/
│   ├── longbench_eval.py       # ★ Main: LongBench SOTA comparison, all policies
│   ├── longbench_metrics.py    # F1, ROUGE-L, substring-match scorers
│   └── plot_sota_comparison.py # Bar chart + radar chart from LongBench JSON
├── sota/
│   ├── snapkv/                 # FasterDecoding/SnapKV (upstream)
│   ├── streamingllm/           # mit-han-lab/streaming-llm (upstream)
│   ├── h2o/                    # FMInference/H2O (upstream)
│   └── quest/                  # mit-han-lab/Quest (upstream)
├── figures/                    # Generated publication figures (PNG/JPG)
├── results/                    # JSON result files (gitignored from build outputs)
├── tests/
│   └── test_tiered_cache.py    # 7 unit tests
├── TieredKV_Project_Explanation.md   # Full project write-up
├── TieredKV_Project_Explanation.pdf  # PDF version with embedded figures
└── requirements.txt
```

---

## Environment Setup

**Requirements:** Python 3.10+, CUDA GPU (tested on NVIDIA L40S, CUDA 12.4, 48 GB VRAM).

```bash
# 1. Create a virtual environment
python -m venv venv_kvcache
source venv_kvcache/bin/activate

# 2. Install PyTorch 2.5+ with CUDA (adjust cu124 to your CUDA version)
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Set Hugging Face cache directory (must have 20+ GB free)
export HF_HOME=/path/to/your/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Download Mistral-7B-Instruct-v0.2

```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir ./models/mistral-7b-instruct \
    --max-workers 8
```

> For Llama-3-8B: accept the license at [huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), run `huggingface-cli login`, then replace the model path.

---

## Quick Verification (No Model Required)

```bash
# Unit tests (10 tests, ~2 seconds)
python -m pytest tests/ -v
```

---

## Reproducing All Results

### Step 1 — Baselines at budget=1024 (Full, StreamingLLM, H2O, SnapKV)

```bash
MODEL=./models/mistral-7b-instruct

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
python experiments/plot_sota_comparison.py \
    --json results/headline_equal_vram.json \
    --outdir figures/
```

Output: `figures/fig_longbench_comparison.png`, `figures/fig_radar_comparison.png` (merge Step 1/2's JSON files into one `config`/`results` file first — see `results/headline_equal_vram.json` for the schema).

---

## Hardware & Cost Model

Calibrated analytically for **NVIDIA L40S** (tested hardware):

| Tier | Technology | Read BW | Write BW | Energy |
|------|-----------|---------|---------|--------|
| Tier 1 — VRAM | GDDR6 | 864 GB/s | 864 GB/s | 12 pJ/bit |
| Tier 2 — STT-RAM | STT-MRAM (simulated) | 1 TB/s | 250 GB/s | 2 / 8 pJ/bit |
| Tier 3 — DRAM | DDR5 | 200 GB/s | 200 GB/s | 20 pJ/bit |

The 4× write/read asymmetry in STT-RAM is exactly what the inclusive victim cache exploits — by reusing shadow backups, it avoids the slow/expensive write path entirely.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 7 unit tests cover: bifurcation, promote/demote cycle, inclusive write savings, capacity invariants, full decode loop, ablation (destructive eviction), and accuracy vs oracle.

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
