# TieredKV: A 3-Tier KV Cache Memory Hierarchy for LLM Inference

A research implementation of a three-tier KV cache memory hierarchy that exploits STT-RAM's read-fast/write-slow asymmetry to recover the accuracy lost by permanent-eviction methods, without the full memory cost of keeping every token resident.

```
GPU VRAM (hot)  ──►  STT-RAM victim cache (warm)  ──►  DRAM / Drop (cold)
```

**Key idea:** when a token is promoted from STT-RAM to VRAM, its STT-RAM copy is kept as an immutable shadow backup. A later re-demotion reactivates the backup at zero write cost. Because KV tensors never change after prefill, the backup is always valid. This collapses 97% of STT-RAM write traffic while preserving full accuracy.

---

## How It Works

Every decode step runs four phases:

1. **Sketch check** — Quest-style min/max upper-bound scoring over STT-RAM pages to find promotion candidates cheaply.
2. **Promote** — Top-scored pages move from STT-RAM → VRAM.
3. **Attention** — Exact scaled dot-product attention over the VRAM working set only.
4. **Evict + demote** — Lowest cumulative-attention VRAM token demotes to STT-RAM (inclusive backup reused if available).

Attention sinks (positions 0..3) and the recent sliding window are never evicted.

---

## Techniques Combined

| Technique | Paper | Role in TieredKV |
|---|---|---|
| SnapKV | Li et al., NeurIPS 2024 | Prompt bifurcation scoring |
| StreamingLLM | Xiao et al., ICLR 2024 | Sink + window pinning |
| Quest | Tang et al., ICML 2024 | STT-RAM page sketch scoring |
| H2O | Zhang et al., NeurIPS 2023 | Cumulative-attention eviction |
| FlexGen | Sheng et al., ICML 2023 | DRAM offload path |

---

## Repository Structure

```
KV_Cache_Optimization/
├── src/
│   ├── attention.py        # Shared SDPA math + MAC counting
│   ├── tiered_kv_cache.py  # Core 3-tier cache implementation
│   ├── baselines.py        # FullAttention, StreamingLLM, H2O, SnapKV, Quest
│   ├── cost_model.py       # Analytical latency/energy (L40S calibrated)
│   └── metrics.py          # Per-step and aggregate metrics
├── experiments/
│   ├── model_wrapper.py    # Real model evaluation (SDPA hook, Mistral/Llama)
│   ├── compare_accuracy.py # Synthetic 5-way baseline comparison
│   ├── sweep_budgets.py    # Sweep Tier 1 (VRAM) budget sizes
│   ├── sweep_window.py     # Sweep sliding window sizes
│   ├── plot_optimization.py # Plots optimal VRAM + window size
│   └── plot_results.py     # General result visualisation
├── sota/
│   ├── snapkv/             # FasterDecoding/SnapKV (upstream)
│   ├── streamingllm/       # mit-han-lab/streaming-llm (upstream)
│   ├── h2o/                # FMInference/H2O (upstream)
│   └── quest/              # mit-han-lab/Quest (upstream)
├── tests/
│   └── test_tiered_cache.py
└── results/                # Output JSON files (gitignored)
```

---

## Environment Setup

**Requirements:** Python 3.10+, CUDA GPU (tested on NVIDIA L40S with CUDA 12.4).

```bash
# 1. Create a venv in your workspace directory
python -m venv venv_kvcache
source venv_kvcache/bin/activate

# 2. Install PyTorch 2.5+ with CUDA (adjust cu124 to your CUDA version)
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# 3. Install other dependencies
pip install transformers accelerate datasets huggingface_hub numpy matplotlib pytest

# 4. Download Mistral-7B-Instruct (instruction-tuned for QA)
#    Set HF_HOME to a directory with 20+ GB free (e.g. under /data/)
export HF_HOME=$PWD/hf_cache
hf download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir $PWD/models/mistral-7b-instruct \
    --max-workers 8
```

> For Llama-3-8B: accept the license at [huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), then run `hf auth login` and replace the model path below.

---

## Quick Verification (no model required)

```bash
cd KV_Cache_Optimization

# Runs full code path with synthetic tensors — no GPU, no model download
python experiments/model_wrapper.py --smoke

# Unit tests (7 tests, ~5 seconds)
python -m pytest tests/ -v
```

---

## Reproducing All Results

### Step 1 — Real model evaluation

```bash
# Set MODEL to the directory where you downloaded Mistral-7B-Instruct.
# If you followed the setup steps, it is at $PWD/../models/mistral-7b-instruct
MODEL=$PWD/../models/mistral-7b-instruct

python experiments/model_wrapper.py \
    --model $MODEL \
    --prompt-len 512 \
    --sram 128 --stt 256 \
    --device cuda \
    --json results/mistral7b_real.json
```

Expected output (32-layer aggregate):
```
mean Acc(all)=0.9753  compute saved 65.6%  write savings 97.0%
```

---

### Step 2 — Sweep Tier 1 (VRAM) budget

Find the accuracy/compute trade-off across different VRAM budgets.

```bash
MODEL=$PWD/../models/mistral-7b-instruct

python experiments/sweep_budgets.py \
    --model $MODEL \
    --prompt-len 512 --stt 256 \
    --device cuda
```

Results are saved to `results/sweep_vram_*.json`.

---

### Step 3 — Sweep sliding window size

```bash
MODEL=$PWD/../models/mistral-7b-instruct

python experiments/sweep_window.py \
    --model $MODEL \
    --prompt-len 512 \
    --sram 128 --stt 256 \
    --windows 4 8 16 32 64 128 \
    --device cuda \
    --json results/window_sweep.json
```

---

### Step 4 — Generate optimisation graphs

Produces two publication-quality figures showing the optimal VRAM budget and optimal window size across all three metrics (accuracy, compute saved, write savings).

```bash
pip install matplotlib  # if not already installed

python experiments/plot_optimization.py \
    --results-dir results \
    --window-json results/window_sweep.json \
    --outdir figures/
```

Output: `figures/fig_optimal_vram.png`, `figures/fig_optimal_window.png`


### Step 5 — LongBench SOTA comparison (the main result)

Runs all 5 methods (Full, StreamingLLM, H₂O, SnapKV, TieredKV) on LongBench QA tasks using the **same model, same budget, same scoring**. This is the fair head-to-head comparison.

```bash
MODEL=$PWD/../models/mistral-7b-instruct

python experiments/longbench_eval.py \
    --model $MODEL \
    --tasks multifieldqa_en narrativeqa passage_retrieval_en \
    --methods full streamingllm h2o snapkv tieredkv \
    --budget 384 \
    --device cuda \
    --json results/longbench_comparison.json
```

> **Note:** Running all methods on all samples takes ~2-4 hours. Use `--max-samples 20` for a quick preview.

---

### Step 6 — SOTA comparison plots and tables

Generates bar chart, radar chart, and LaTeX table from the LongBench results.

```bash
python experiments/plot_sota_comparison.py \
    --json results/longbench_comparison.json \
    --outdir figures/
```

Output: `figures/fig_longbench_comparison.png`, `figures/fig_radar_comparison.png`, plus LaTeX table on stdout.

---

## Key Results (Mistral-7B-v0.1, 512-token context, VRAM=128, STT=256)

| Metric | Value |
|---|---|
| Accuracy vs oracle (cosine) | **0.9753** |
| Compute saved | **65.6%** |
| STT-RAM write savings (inclusive cache) | **97.0%** |
| Peak VRAM occupancy | 128 tokens |
| Peak STT-RAM occupancy | 256 tokens |

### Optimal VRAM budget (from sweep)

| VRAM tokens | Accuracy | Compute (GOPs) | STT Writes (Tokens) |
|---|---|---|---|
| 32 | 0.9440 | 0.042 (Oracle: 0.268) | 32 paid / 1056 total |
| 64 | 0.9594 | 0.059 (Oracle: 0.268) | 32 paid / 1056 total |
| 128 | 0.9753 | 0.092 (Oracle: 0.268) | 32 paid / 1056 total |
| 192 | 0.9811 | 0.126 (Oracle: 0.268) | 32 paid / 1056 total |
| 256 | 0.9845 | 0.159 (Oracle: 0.268) | 32 paid / 1052 total |
| 384 | 0.9888 | 0.222 (Oracle: 0.268) | 32 paid / 1037 total |

**Recommended:** 128 tokens (25% of context) — best balance across all three metrics.

### Optimal sliding window (from sweep)

| Window | Accuracy | Compute saved | Write savings |
|---|---|---|---|
| 4 | 0.9747 | 97.9% | 92.9% |
| **16** | **0.9753** | **97.9%** | **97.0%** |
| 32 | 0.9747 | 97.9% | 97.0% |
| 64 | 0.9758 | 97.9% | 97.0% |

**Recommended:** 16 tokens — minimum size that achieves 97% write savings.

---

## Hardware & Cost Model

Calibrated for **NVIDIA L40S** (tested hardware):

| Tier | Technology | Bandwidth | Energy |
|---|---|---|---|
| Tier 1 (VRAM) | GDDR6 | 864 GB/s | 12 pJ/bit |
| Tier 2 (STT-RAM) | STT-MRAM (simulated) | 1 TB/s read / 250 GB/s write | 2 / 8 pJ/bit |
| Tier 3 (DRAM) | DDR5 | 200 GB/s | 20 pJ/bit |

The 4× write/read asymmetry in STT-RAM is what the inclusive victim cache exploits. Latency and energy are derived analytically from these figures; accuracy and migration counts are measured empirically.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 7 tests cover: bifurcation, promote/demote cycle, inclusive write savings, capacity invariants, full decode loop, ablation (destructive eviction), and accuracy vs oracle.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{kumar2025tieredkv,
  title   = {TieredKV: Exploiting STT-RAM Read/Write Asymmetry for
             Near-Lossless KV Cache Eviction in LLM Inference},
  author  = {Kumar, Prashant},
  year    = {2025},
  url     = {https://github.com/PrashantKumar-07/KV_Cache_Optimization}
}
```
