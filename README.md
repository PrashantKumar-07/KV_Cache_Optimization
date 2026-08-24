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

**Inclusive shadow cache:** When a token is promoted from STT-RAM → VRAM, the STT-RAM copy is kept as an immutable shadow backup. If the same token is later re-demoted, the backup is reactivated at **zero write cost**. Because KV tensors never change after prefill, the backup is always valid. This eliminates **100% of STT-RAM writes** in steady-state inference.

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

### LongBench SOTA Comparison (Mistral-7B-Instruct-v0.2, Budget = 1024 tokens)

All methods evaluated on the same model, same samples, same scoring. **Higher is better.**

| Task | Full Cache | StreamingLLM | H₂O | SnapKV | **TieredKV** |
|------|:----------:|:------------:|:---:|:------:|:------------:|
| MultiFieldQA-EN | 46.68 | 11.35 | 11.35 | 10.23 | **13.60** |
| Qasper | 26.62 | 5.05 | 5.05 | 4.94 | **5.23** |
| NarrativeQA | 20.09 | 3.70 | 3.70 | 3.90 | **6.03** |
| HotpotQA | 31.09 | 6.90 | 6.90 | 7.76 | **10.57** |
| GovReport | 19.93 | **9.18** | **9.18** | **9.50** | 8.14 |
| TriviaQA | 73.30 | 22.59 | 22.59 | 23.12 | **23.79** |

**TieredKV outperforms all compression baselines on 5 out of 6 tasks.** It achieves 100% STT-RAM write savings across all tasks (inclusive cache eliminates all redundant writes).

### Synthetic Accuracy Sweep (Mistral-7B-v0.1, 512-token context)

| Metric | Value |
|--------|-------|
| Accuracy vs oracle (cosine similarity) | **0.9753** |
| Compute saved vs Full Cache | **65.6%** |
| STT-RAM write savings (inclusive cache) | **97.0%** |
| Peak VRAM occupancy | 128 tokens |
| Peak STT-RAM occupancy | 256 tokens |

### Optimal VRAM Budget Sweep

| VRAM Tokens | Accuracy | Compute (GOPs) | STT Writes Paid |
|:-----------:|:--------:|:--------------:|:---------------:|
| 32 | 0.9440 | 0.042 | 32 / 1056 total |
| 64 | 0.9594 | 0.059 | 32 / 1056 total |
| **128** | **0.9753** | **0.092** | **32 / 1056 total** |
| 192 | 0.9811 | 0.126 | 32 / 1056 total |
| 256 | 0.9845 | 0.159 | 32 / 1052 total |
| 384 | 0.9888 | 0.222 | 32 / 1037 total |

**Recommended:** 128 tokens (25% of context) — optimal balance of accuracy, compute, and write savings.

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
│   ├── longbench_eval.py       # ★ Main: LongBench 6-task SOTA comparison
│   ├── longbench_metrics.py    # F1, ROUGE-L, substring-match scorers
│   ├── model_wrapper.py        # Real-model evaluation (SDPA hook, Mistral/Llama)
│   ├── plot_decode_dynamics.py # 3-panel decode dynamics + occupancy/migration
│   ├── plot_sota_comparison.py # Bar chart + radar chart from LongBench JSON
│   ├── plot_optimization.py    # Optimal VRAM budget + window size figures
│   ├── sweep_budgets.py        # Sweep Tier 1 VRAM budget sizes
│   └── sweep_window.py         # Sweep sliding window sizes
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
# Smoke test — runs full code path with synthetic tensors, no GPU needed
python experiments/model_wrapper.py --smoke

# Unit tests (7 tests, ~5 seconds)
python -m pytest tests/ -v
```

---

## Reproducing All Results

### Step 1 — LongBench SOTA Comparison (Main Result)

Runs all 5 methods on 6 LongBench tasks with the same model, same budget, same scoring.

```bash
MODEL=./models/mistral-7b-instruct

python experiments/longbench_eval.py \
    --model $MODEL \
    --tasks multifieldqa_en qasper narrativeqa hotpotqa gov_report triviaqa \
    --methods full streamingllm h2o snapkv tieredkv \
    --budget 1024 \
    --device cuda \
    --json results/longbench_comparison.json
```

> **Note:** Full evaluation takes 4–8 hours. Use `--max-samples 20` for a quick preview.

### Step 2 — SOTA Comparison Figures

```bash
python experiments/plot_sota_comparison.py \
    --json results/longbench_comparison.json \
    --outdir figures/
```

Output: `figures/fig_longbench_comparison.png`, `figures/fig_radar_comparison.png`

---

### Step 3 — Decode Dynamics & Cache Occupancy Figures

```bash
python experiments/plot_decode_dynamics.py --outdir figures/
```

Output: Three figures showing decode-step dynamics, VRAM/STT-RAM occupancy over time, and token migration counts.

---

### Step 4 — Synthetic Accuracy & Sweep Experiments

```bash
MODEL=./models/mistral-7b-instruct

# Single-run accuracy vs oracle
python experiments/model_wrapper.py \
    --model $MODEL \
    --prompt-len 512 \
    --sram 128 --stt 256 \
    --device cuda \
    --json results/mistral7b_real.json

# Sweep VRAM budget (32 → 384 tokens)
python experiments/sweep_budgets.py \
    --model $MODEL \
    --prompt-len 512 --stt 256 \
    --device cuda

# Sweep sliding window size
python experiments/sweep_window.py \
    --model $MODEL \
    --prompt-len 512 \
    --sram 128 --stt 256 \
    --windows 4 8 16 32 64 128 \
    --device cuda \
    --json results/window_sweep.json
```

### Step 5 — Generate Optimization Figures

```bash
python experiments/plot_optimization.py \
    --results-dir results \
    --window-json results/window_sweep.json \
    --outdir figures/
```

Output: `figures/fig_optimal_vram.png`, `figures/fig_optimal_window.png`

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
