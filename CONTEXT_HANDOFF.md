# KV-Cache 3-Tier Memory Hierarchy — Context Handoff

Paste this whole file into a new chat to restore full context. It captures the research goal, decisions, file interfaces, and current status. Nothing else is needed.

---

## 1. Who I am & what you should be

I'm an academic researcher writing a paper. **Act as a senior research mentor / computer-architecture expert.** Give concrete, quantitative answers: math, dimension tables, token-by-token traces, measured numbers — never vague prose or hand-waving "assumptions."

**Hard preference (I corrected this once):** when I ask about architecture/data-flow, give a *granular step-by-step trace* (actual tensors, shapes, attention-weight numbers), not a conceptual summary. "Detail that contains nothing" gets rejected.

Build **your own** implementation. Don't defer to any other LLM's plan.

---

## 2. The research idea

A **3-tier KV-cache memory hierarchy for LLM inference**:

```
SRAM (fast, small, leaky)  →  STT-RAM (fast-read / slow+costly-write, ~0 leakage, dense)  →  DRAM / Drop
```

- Hot KV tokens live in **SRAM**.
- Evicted tokens fall to **STT-RAM as a VICTIM CACHE** — read-fast so they can be promoted back cheaply.
- Cold tokens spill to **DRAM** or are dropped.

**Novelty claim:** first to exploit STT-RAM's read-fast/write-slow asymmetry as a victim cache for KV tokens, enabling **near-lossless eviction** — recovering the accuracy that permanent-eviction methods (SnapKV, H2O) throw away. Attention math stays IDENTICAL everywhere; only *which K/V tokens are resident* changes.

**Techniques combined:** SnapKV (prompt compression), Quest (min/max sketch routing), StreamingLLM (attention sinks + sliding window), H2O (cumulative-attention importance for eviction), InfiniGen (async prefetch), FlexGen (DRAM offload).

---

## 3. Workflow (important — I can't run Claude Code on my server)

1. You give me full, self-contained, argparse-driven code locally.
2. I push to GitHub.
3. I copy the repo onto **Server 3** and run the full model there.
4. I bring bugs/queries/updates back here, then manually copy-paste updated file contents into the server's destination files.

So: **everything must be portable, dependency-light, argparse-driven, and runnable both locally (synthetic/smoke) and on the server (real model).**

Local eval is on small/synthetic data. Server does the real run: **Llama-3-8B on LongBench**. Evaluate everything papers report: size, latency, accuracy, GOPs, MAC ops, token/embedding size, occupancy, migration volume — real facts, not assumptions.

---

## 4. Transferability principle (settled)

The compute/memory trade-off **shape** is architectural and transfers from synthetic → real. But **accuracy at each budget** and **migration volume** depend on the real attention distribution and do **NOT** transfer numerically. Hence Option A (real model) exists alongside Option B (synthetic sweep). Every experiment prints a "transfer reminder."

---

## 5. Environment

- Windows 10, Python 3.14.6, torch 2.13.0+cpu, numpy 2.4.4, **no CUDA locally**, shell = bash.
- Project root: `C:\Users\pegas\kv_cache`.
- Git repo initialized; pushed to `https://github.com/PrashantKumar-07/KV_Cache_Optimization.git`.

---

## 6. Key technical facts / conventions

- Reference transformer: d_model=512, h=8, d_k=d_v=64, d_ff=2048, N=6.
- **Llama-3-8B:** 32 layers, 32 query heads, head_dim=128, GQA with 8 KV heads.
- **Tensor convention everywhere:** `(num_heads H, seq_len, head_dim D)`. Single decode query = `(H, 1, D)`.
- Attention = `softmax(QKᵀ/√d_k)·V`, kept identical across Full/Streaming/SnapKV/Tiered.
- Attention sinks: tokens `0..sink_size-1` pinned. Sliding window: recent `window_size` tokens pinned.
- Quest sketch: per-page `max(Q·min_k, Q·max_k)` upper-bound score → cheap routing (~8× less traffic).
- SnapKV importance: observation-window queries × all keys, avg-pooled (kernel=5).
- H2O: cumulative attention score for eviction; LRU for STT→DRAM/drop.
- **Measured empirically:** correctness, accuracy (cosine vs oracle), MACs/GOPs, tier occupancy, migration counts.
- **Derived analytically:** latency & energy (NVSim/CACTI/Destiny-style per-bit model). Never conflate the two.
- **Crossover point** = smallest SRAM budget where TieredKV is faster than full attention AND accuracy stays acceptable.

### SDPA-hook capture (Option A's core trick)
HF Llama calls `torch.nn.functional.scaled_dot_product_attention` exactly once per decoder layer, **after RoPE** on Q/K and **after GQA `repeat_kv`** expands KV to 32 heads. Wrapping SDPA for one forward pass records exact post-RoPE/post-repeat `(Q,K,V)` of shape `(batch,32,seq,128)` in layer order — version-independent, no internals patching. Routing at query-head granularity (H=32) gives EXACT attention. GQA-aware per-group sharing (up to 4× less KV memory) is a future *memory* optimization, not an accuracy change.

---

## 7. Files (all under `C:\Users\pegas\kv_cache`)

### `src/cost_model.py` — DONE
Analytical latency/energy. `TierSpec(name, read_bw_gbps, write_bw_gbps, read_pj_per_bit, write_pj_per_bit, leakage_mw_per_mb)`.
`DEFAULT_TIERS`:
- SRAM: 19500/19500 GB/s, 1/1 pJ/bit, leak 80
- STT-RAM: 1000 read / 250 write GB/s, 2/8 pJ/bit, leak 1
- DRAM: 200/200 GB/s, 20/20 pJ/bit, leak 15

`CostModel(tiers=DEFAULT_TIERS, bytes_per_elem=2)` methods: `read/write_latency_us`, `read/write_energy_nj`, `promote_cost` (STT read + SRAM write), `demote_cost` (SRAM read + STT write), `deep_demote_cost` (STT read + DRAM write), `sram_compute_read_cost`. Migration methods return `(latency_us, energy_nj)`.

### `src/metrics.py` — DONE
`StepRecord`: step, attn_macs, sketch_macs, promoted/demoted/dropped_tokens, sram/sttram/dram_tokens, latency_us, energy_nj, lat_sketch/promote/attention/demote_us.
`RunMetrics(label, steps, num_heads, head_dim, prompt_len)`: `.add(rec)`, props total_macs, total_gops (`macs*2/1e9`), total_promoted/demoted/dropped, total_latency_us, total_energy_nj, peak_sram/sttram_tokens; `kv_bytes(tokens, bpe=2)=tokens*2*num_heads*head_dim*bpe`; `summary()`; `to_json(path)`.

### `src/tiered_kv_cache.py` — DONE
`TieredConfig(num_heads=8, head_dim=64, sink_size=4, window_size=16, sram_capacity=64, sttram_capacity=128, page_size=16, promote_top_pages=2, store_dram=False, pool_kernel=5, dtype, device)`.
`TieredKVCache(cfg, cost_model=None)`: `initial_bifurcation(k_all, v_all, q_all)` → occupancy dict; `step(q, new_k, new_v, new_pos)` → `(H,1,D)`; exposes `.metrics` (RunMetrics).

### `src/baselines.py` — DONE
`FullAttention` (oracle): `reset_prompt`, `step`, `total_macs`, `num_tokens`.
`StreamingLLM(sink_size, window_size)`, `SnapKV(budget, sink_size, window_size, pool_kernel)`.

### `src/attention.py` — DONE
`scaled_dot_product_attention(q,k,v,causal=False)`; `snapkv_importance(q_window, k_all, pool_kernel=5)`.

### `tests/test_tiered_cache.py` — DONE
5 plain-assert tests, all passing.

### `experiments/sweep_sram.py` — DONE & VERIFIED (Option B)
Finds the SRAM-budget crossover. Args: `--budgets` (default 16,32,64,96,128,192,256), `--prompt` 512, `--steps` 150, `--heads` 8, `--dim` 64, `--stt-mult` 2.0, `--acc-thresh` 0.30, `--device` cpu, `--csv`.
`make_scenario(H,D,prompt_len,num_steps,anchor_frac=0.35,scale=3.0,device,seed=42)` (CPU generator then `.to(device)`, injects long-range anchors). `run_full`, `run_tiered`. Verdict "WIN" if net>0 AND acc_all≥thresh; tracks first crossover; prints table + crossover + transfer reminder + optional CSV.
**Result on synthetic data:** no crossover in swept range (all "slow"); accuracy rises monotonically with SRAM budget (Acc_anch 0.0276 → 0.9587 as SRAM 16→256). Expected — real number must come from server.

### `experiments/compare_accuracy.py` — DONE
Head-to-head Full/StreamingLLM/SnapKV/TieredKV. H=8, D=64, prompt_len=512, steps=150, SRAM=64, STT=128. StreamingLLM window=SRAM. Splits Acc(all)/Acc(anchor)/Acc(local). Saves `results/tiered_run.json`.

### `experiments/model_wrapper.py` — DONE & VERIFIED (Option A)
Runs the 3-tier cache on REAL (or synthetic-smoke) per-layer attention tensors.
- `SDPACapture` context manager patches `torch.nn.functional.scaled_dot_product_attention` for one forward pass → records post-RoPE / post-`repeat_kv` `(q,k,v)` at `(batch,32,seq,128)` in layer order. `capture_layers(model, input_ids)` squeezes batch row 0 → per-layer `(H,seq,D)`.
- `synthetic_layers(...)` fabricates per-layer `(q,k,v)` + per-position anchor masks with the SAME long-range structure as `compare_accuracy` — powers `--smoke` with **no transformers / no model download**.
- `real_layers(model_name, text, prompt_len)` lazily imports transformers, loads an HF causal LM (`attn_implementation="sdpa"`, fp16), captures one prefill. masks=None (no ground-truth anchor labels on real text).
- `eval_layer(qkv, split, sram, stt, flags)` replays one layer through `FullAttention` oracle + `TieredKVCache` (prompt = first `split` tokens, rest streamed one-at-a-time); returns measured dict (acc_all/anchor/local cosine, tiered vs oracle GOPs, promoted/demoted/dropped, peak SRAM/STT tokens+KB, derived latency/energy).
- `report(per_layer, meta)` prints per-layer table + aggregate, returns aggregate dict; every run prints the transfer reminder.
- argparse `main()`: `--smoke`, `--model`, `--text-file`, `--prompt-len`, `--sram`, `--stt`, `--layers`, `--smoke-layers`, `--smoke-seq`, `--heads`, `--dim`, `--device`, `--json`.
**Verified:** `python experiments/model_wrapper.py --smoke` runs the full path (synthetic_layers → eval_layer → report → optional JSON) end-to-end with no transformers install and no AttributeErrors.
Server run (real weights):
```
python experiments/model_wrapper.py --model meta-llama/Meta-Llama-3-8B \
    --text-file prompt.txt --prompt-len 256 --sram 128 --stt 256 \
    --layers 0 1 2 3 --json results/llama_layers.json
```

### `requirements.txt` — torch>=2.13.0, numpy>=2.0
### `.gitignore` — `__pycache__/`, `*.pyc`, `results/*.json`, `.venv/`, `venv/`, `.DS_Store`
### `README.md` — DONE — architecture-mapping table, decode loop, measured-vs-derived, cost-model constants, next steps, known simplifications.

---

## 8. Status summary

- Option B (sweep) — complete, verified.
- Baselines, cache, cost model, metrics, tests — complete.
- **Option A (`model_wrapper.py`) — COMPLETE & smoke-verified.** All local code is done.
- Repo pushed to GitHub: `https://github.com/PrashantKumar-07/KV_Cache_Optimization.git`.
- **Immediate remaining work is on the SERVER, not here:** run `model_wrapper.py` with real Llama-3-8B weights (drop `--smoke`) on Server 3, and run the full LongBench evaluation. The synthetic/smoke numbers are trade-off SHAPE only — real accuracy-at-budget and migration VOLUME must come from that run.
- Optional: README update with server data-capture guidance.

---

## 9. Immediate next step for the new chat

All local code is complete and pushed. The next step is on **Server 3**, not here: run the real Llama-3-8B capture (`experiments/model_wrapper.py` without `--smoke`) and the full LongBench evaluation, then bring real accuracy-at-budget and migration-volume numbers back here. See section 8.
