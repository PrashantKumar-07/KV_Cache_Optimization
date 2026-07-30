# 3-Tier KV-Cache Memory Hierarchy (SRAM → STT-RAM → DRAM)

A pure-PyTorch, hardware-agnostic proof-of-concept for query-aware routing of
Transformer KV-cache tokens across a 3-tier memory hierarchy. The central idea:
instead of **permanently deleting** low-importance tokens (SnapKV / H2O), demote
them to an on-chip **STT-RAM victim cache** whose fast-read / slow-write
asymmetry matches the write-once / read-maybe lifecycle of a KV token, then
**promote them back** when a later query needs them.

This repo is the *local, small-scale* stage: prove correctness + the accuracy
benefit on tiny synthetic workloads with full instrumentation, before running
real 8B models on LongBench on the server.

---

## Layout

```
src/
  attention.py        scaled-dot-product attention + SnapKV importance + MAC counting
  cost_model.py       analytical latency/energy model (SRAM/STT-RAM/DRAM per-bit figures)
  metrics.py          per-step + aggregate metrics (MACs, GOPs, traffic, occupancy)
  tiered_kv_cache.py  the core 3-tier router (Phase-0 bifurcation + 4-step decode loop)
  baselines.py        FullAttention (oracle), StreamingLLM, SnapKV
tests/
  test_tiered_cache.py   invariant tests (sinks pinned, window kept, bounds held, promotion works)
experiments/
  compare_accuracy.py    Full vs StreamingLLM vs SnapKV vs TieredKV on a long-range scenario
results/
  tiered_run.json        per-step metrics dump (created on run)
```

## Run

```bash
pip install -r requirements.txt
python tests/test_tiered_cache.py        # correctness
python experiments/compare_accuracy.py   # head-to-head accuracy + migration stats
```

---

## The architecture (what maps to what)

| Transformer concept | This repo | Source idea |
|---|---|---|
| Full KV cache | `TieredKVCache` (3 tiers combined) | — |
| Hot working set | Tier 1 SRAM (`sram_k/v`) | StreamingLLM |
| Warm victim cache | Tier 2 STT-RAM (`stt_k/v`, paged sketches) | **our novelty** |
| Cold storage / drop | Tier 3 DRAM (`dram_k/v`) or discard | H2O / FlexGen |
| Prompt compression | `initial_bifurcation` | SnapKV |
| Importance score | `sram_cum` (cumulative attention) | H2O |
| Sketch-based routing | `sketch_check` (min/max upper bound) | Quest |
| Sink + window pinning | `_protected_mask` | StreamingLLM |

The attention **math is never modified** — only which K/V tokens are physically
resident changes. Any accuracy gap is a pure memory-management effect.

## Decode loop (every generated token)

1. **`sketch_check`** — min/max upper-bound scores over STT-RAM pages (cheap,
   ~8× less traffic than scanning all keys) → which pages look useful.
2. **`promote`** — move predicted-useful pages STT-RAM → SRAM.
3. **`compute_attention`** — exact attention over the SRAM working set only.
4. **`evict_and_demote`** — SRAM→STT-RAM (lowest cumulative attention, sinks &
   window protected); STT-RAM→DRAM/drop (LRU) when the warm tier overflows.

---

## Measured vs derived (be honest with reviewers)

- **Measured empirically** (real, machine-independent): correctness, accuracy
  vs the full-attention oracle (cosine similarity), MACs / GOPs, tier occupancy,
  promotion / demotion / drop counts.
- **Derived analytically** from `cost_model.py`: latency (µs) and energy (nJ).
  STT-RAM is not a physical device here (or on the server), so — exactly as
  NVSim / CACTI / Destiny-based papers do — we cost every moved byte with
  calibrated per-bit figures. These constants are **parameters to sweep**, not
  ground truth. Edit `DEFAULT_TIERS` in `cost_model.py` to match whatever device
  model your paper cites.

### Cost-model constants (defaults, representative NVM figures)

| Tier | Read BW | Write BW | Read energy | Write energy |
|---|---|---|---|---|
| SRAM | 19.5 TB/s | 19.5 TB/s | 1 pJ/bit | 1 pJ/bit |
| STT-RAM | 1.0 TB/s | 250 GB/s | 2 pJ/bit | 8 pJ/bit |
| DRAM | 200 GB/s | 200 GB/s | 20 pJ/bit | 20 pJ/bit |

The physics the design exploits: STT-RAM **read ≈ 4× faster & cheaper than its
write**, and its **near-zero leakage** — so evicting (write-once) is tolerable
while promoting (read-often) is cheap.

---

## Interpreting `compare_accuracy.py`

- `Acc(anchor)` = accuracy on decode steps with an injected long-range
  dependency. This is where permanent eviction fails and the victim cache wins.
- `Acc(local)` = accuracy on ordinary local steps (all methods similar).
- Absolute values are synthetic (random Gaussians → low cosine baseline). **The
  ordering — TieredKV > SnapKV > StreamingLLM on `Acc(anchor)` — is the claim.**

---

## Next steps (server, full scale)

1. Replace synthetic Q/K/V with a real model (Llama-3-8B) — feed per-layer,
   per-head K/V from HuggingFace into `TieredKVCache`, one cache per layer.
2. Swap the synthetic scenario for **LongBench** subtasks (multi-hop QA,
   retrieval) where long-range recall is real.
3. Move tensors to CUDA (`TieredConfig(device="cuda")`), add `torch.cuda.Event`
   wall-clock timing alongside the analytical model as a sanity cross-check.
4. Sweep budgets (SRAM/STT capacities, `page_size`, `promote_top_pages`) to find
   the crossover where attention savings exceed migration overhead.
5. Calibrate `cost_model.py` constants to the exact STT-RAM device your paper cites.

## Known simplifications (POC scope)

- Single attention layer per cache instance (wrap per-layer for a full model).
- STT-RAM pages are contiguous chunks recomputed on the fly (fine at small
  scale; a real kernel would keep sketches incrementally).
- Promotion is currently aggressive (`promote_top_pages=2` with a loose
  threshold) — tune for the compute/accuracy trade-off you want to report.
