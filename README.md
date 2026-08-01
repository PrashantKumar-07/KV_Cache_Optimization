# 3-Tier KV-Cache Memory Hierarchy (SRAM → STT-RAM → DRAM)

A pure-PyTorch, hardware-agnostic proof-of-concept for query-aware routing of
Transformer KV-cache tokens across a 3-tier memory hierarchy. The central idea:
instead of **permanently deleting** low-importance tokens (SnapKV / H2O), demote
them to an on-chip **STT-RAM victim cache** whose fast-read / slow-write
asymmetry matches the write-once / read-maybe lifecycle of a KV token, then
**promote them back** when a later query needs them.

The victim cache alone would pay a heavy price on every re-eviction: an STT-RAM
**write** is ~4× slower and costlier than a read. Because KV tokens are
**immutable after prefill**, we make the victim cache **inclusive** — a promoted
token's STT copy is kept as a clean *shadow backup*, so a later re-demote just
reactivates the backup at **zero write cost**. On a 150-step thrashing scenario
this eliminates 96.8% of demotion writes and cuts derived latency 3.7× (54.4 →
14.6 µs) with **identical accuracy**. See "Migration overhead" below.

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
  test_tiered_cache.py   invariant tests (sinks pinned, window kept, bounds held, promotion works, inclusive cache saves writes)
experiments/
  compare_accuracy.py    Full vs StreamingLLM vs SnapKV vs TieredKV on a long-range scenario
  sweep_sram.py          SRAM-budget sweep → crossover point (CSV output)
  model_wrapper.py       run the 3-tier cache on real (or --smoke synthetic) per-layer K/V
  plot_results.py        LOCAL-only matplotlib figures (quarantined; not in the server pipeline)
results/
  tiered_run.json        per-step metrics dump (created on run)
```

## Run

```bash
pip install -r requirements.txt
python tests/test_tiered_cache.py                       # correctness (7 tests)
python experiments/compare_accuracy.py                  # inclusive victim cache (default)
python experiments/compare_accuracy.py --no-inclusive \
    --json results/tiered_run_destructive.json          # ablation: destructive eviction
python experiments/plot_results.py \
    results/tiered_run.json results/tiered_run_destructive.json --outdir figures
```

The two `tiered_run*.json` dumps rendered side by side show the STT-write (red)
latency band collapse under the inclusive cache — the visual proof of the
migration-overhead contribution.

---

## The architecture (what maps to what)

| Transformer concept | This repo | Source idea |
|---|---|---|
| Full KV cache | `TieredKVCache` (3 tiers combined) | — |
| Hot working set | Tier 1 SRAM (`sram_k/v`) | StreamingLLM |
| Warm victim cache | Tier 2 STT-RAM (`stt_k/v`, paged sketches, inclusive shadow backups) | **our novelty** |
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
   window protected); reactivate a shadow backup instead of writing when one
   exists (inclusive cache); STT-RAM→DRAM/drop (LRU, backups reclaimed first)
   when the warm tier overflows.

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

## Migration overhead: the inclusive victim cache

A victim cache only helps if promoting/demoting is cheaper than the recall it
buys. The cost is dominated by the **STT-RAM write** (250 GB/s, 8 pJ/bit — vs
1 TB/s, 2 pJ/bit for a read), and thrashing re-demotes the same tokens over and
over. We attack this on two axes:

- **Cost per migration — inclusive cache (`inclusive=True`, default).** KV is
  immutable after prefill, so when `promote` moves a token STT→SRAM we keep its
  STT row as a clean **shadow backup** (`stt_shadow=True`) instead of deleting
  it. A later `evict_and_demote` of that token finds the backup
  (`_find_backup`), flips it live, and pays **no write** (`writes_saved++`).
  Backups are reclaimed before any real victim is dropped, so they never cost
  accuracy. Set `inclusive=False` (or `--no-inclusive`) for the destructive
  ablation.
- **Volume — `promote_top_pages`.** Fewer promotions per step ⇒ less thrashing
  ⇒ fewer demotions to pay for. Exposed as `--promote-top-pages` on the sweep.

**A/B (compare_accuracy, 150 steps, SRAM=64, STT=128), identical accuracy:**

| | demotions | paid writes | writes saved | total latency | total energy |
|---|---|---|---|---|---|
| destructive | 4950 | 4950 | 0 | 54.4 µs | 1.20 mJ |
| **inclusive** | 4950 | **159** | **4791 (96.8%)** | **14.6 µs** | **0.49 mJ** |

**Why not prefetch?** A prefetch hit only hides the *cheap* read; a prefetch
miss wastes a read and can evict a good token, forcing an extra *expensive*
write — increasing the very overhead we reduce. So prefetch is deliberately
excluded; the inclusive cache is the sole cost lever.

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
