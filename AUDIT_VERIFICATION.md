# TieredKV — Independent Verification of the Prior Audit

**Scope:** every claim in the previous LLM audit, re-tested against the code and the result files, plus a full independent pass over the repository.
**Method:** all findings below were reproduced by execution (PyTorch scripts, the real test suite, the real GPU, and direct parsing of the result JSONs), not by reading alone.
**Environment:** NVIDIA L40S ×2, `venv_kvcache`, transformers 4.x, datasets 2.19.2. Test suite: **10 passed in 2.91 s**.

---

## 0a. Remediation status

Everything in §6 that is a code or documentation change has been applied and is
covered by tests. What remains needs GPU re-runs, which are a compute decision
rather than an engineering one.

| §6 item | Status | Evidence |
|---|---|---|
| 3 — `max_ctx` | **Done, with a correction to this audit** | Default raised 8192 → 31500, exposed as `--max-ctx`, recorded in the config and fingerprint. **Raising it alone crashes the tool** — see §0b. Fixed by running prefill under SDPA. Verified end to end on a 31500-token `narrativeqa` sample. |
| 5 — withdraw the false claims | **Done** | "100% write savings" and "improved accuracy" both removed and replaced with the measured per-task figures |
| 6 — fingerprint `max_samples`/`tasks` | **Partly** | Fingerprint extended with `max_samples`, `tasks`, `max_ctx`, `seed`, `tiered_kv_dtype`, and verified to refuse a mismatched resume. **The `frac` sweep still needs re-running at one sample count.** |
| 7 — README architecture table | **Done** | Phases now describe attention-driven promotion and two-tier attention, with a note on why the Quest sketch is not on the harness path |
| 8 — tensor residency state | **Done** | `stt_shadow`, `sram_grace`, `stt_was_vram_resident` are CPU `torch.bool` tensors. `evict_and_demote` 2.55 → 2.12 ms/call; `torch.tensor` construction cost down 73% |
| 9 — `src/baselines.py` | **Done** | `H2O.reset_prompt` honours its budget, both eviction loops guard the all-`inf` argmin, `Quest._sketch_score` bound corrected, scope docstring added |
| 10 — sketch bound | **Done** | Fixed in both files, extracted as `TieredKVCache.page_upper_bounds` so it is directly testable, property test added and **mutation-verified** |
| 11 — bifurcation capacity guard | **Done** | Raises `ValueError` when `sink_size + window_size > sram_capacity` |
| 12 — assorted | **Done** | `datasets` added; test counts reconciled; `CostModel.drop_cost` replaces the DRAM overcharge when `store_dram=False`; `leakage_energy_nj` implemented; `peeked` renamed to `promotions_deferred` |
| 13 — harness integration tests | **Done** | `tests/test_longbench_policies.py`, 11 tests against a tiny Mistral |
| §4.7 — fp32 footprint | **Done (opt-in)** | `TieredConfig.kv_dtype` and `--tiered-kv-dtype` separate K/V storage from score precision; `bfloat16` makes the resident footprint match the equal-VRAM claim. Default unchanged so no existing result silently moves |
| §4.10 — `_visible_tuple` starvation | **Done, and widened** | Counted as `stt_starved_steps`, reported in `stats()`, warned on first occurrence, and mutation-tested. Testing it revealed that hard starvation is rare (a demotion repopulates the live set each step); the real degradation is throttling to the cross-layer minimum, so `stats()` now also reports `mean_stt_exposed` against `stt_expose_quota`. |
| §4.10 — monotonicity, sweep scripts | **Done** | README states the per-task non-monotonicity; sweep scripts default to the 3 tasks and 25 samples actually reported, and pin `--seed` |
| **1 — re-run headline and ablations** | **Open** | Needs hours of GPU. README now carries a prominent notice that its tables are not reproducible with this code |
| **2 — pick the headline comparison** | **Open** | Depends on item 1 |
| **4 — report variance** | **Open** | Depends on item 1. `--seed` is now plumbed and recorded so runs are identifiable |

Test suite: **42 passing**, up from 10. Four regression guards were
mutation-tested — reverting the sketch bound, the stats accumulator fold, the
exposure cap, or the starvation check each makes a specific test fail. The
first version of the sketch test did **not** catch its mutation, because it
recomputed the bound locally instead of calling the production path; that is
why `page_upper_bounds` was extracted.

---

## 0b. Correction to this audit: raising `max_ctx` is not a one-line change

§4.2 recommends raising `max_ctx` from 8192 to the reference protocol's 31500.
That recommendation was incomplete, and applying it literally **breaks the
tool**. Verified by running it:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 59.14 GiB.
GPU 0 has a total capacity of 44.52 GiB
  ...in eager_attention_forward:
  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
```

The model is loaded with `attn_implementation="eager"` because H2O, SnapKV and
TieredKV all need `output_attentions=True` during decode, and only the eager
path produces attention weights. Eager materialises the full
`(batch, heads, q_len, k_len)` score matrix. At decode `q_len` is 1, so that
tensor is a few hundred KB. At **prefill** `q_len` is the whole prompt, so at
31500 tokens it is `32 heads x 31500^2 x 2 bytes = 59.1 GiB` in one
allocation. The 8192 default was therefore load-bearing, not merely
conservative — which the audit did not identify.

The fix is that prefill never requests attention weights, so it does not need
the eager path. `_attn_impl` switches the model to SDPA for the prefill
forward and back to eager for the decode loop. Verified equivalent, not just
non-crashing:

| check | result |
|---|---|
| max abs difference in prefill logits, eager vs SDPA | `0.000e+00` |
| max abs difference in cached K | `0.000e+00` |
| max abs difference in cached V | `0.000e+00` |
| implementation restored after the context manager | `eager` |

K and V come from the projections, which are upstream of the attention
computation, so the cache is bit-identical by construction. With the split in
place, a 31500-token `narrativeqa` sample completes for both H2O and TieredKV.

---

## 0. Bottom line

The prior audit is **roughly 60% correct on facts and substantially wrong on severity**. Its six "Critical" findings are all real bugs, but **four of the six sit in code that has never run** and **one of the six is a misreading of the design**. Meanwhile it missed the single most consequential problem in the repository.

The headline problem is not any of the bugs the audit lists. It is this:

> **Every published number in `README.md` was produced by a version of `experiments/longbench_eval.py` that no longer exists.** The `stt_expose_quota` cap was added to the working tree at 18:31 on 2026-09-07. `README.md` (13:29), `results/headline_equal_vram.json` (13:23), and the entire `results/sweep_stt/` ablation (13:14–13:28) all predate it. Running the README's own reproduction command today produces an average of **~35.3**, not the **38.91** in the README's table.

One correction that matters practically: **the audit's recommended fix #4 (move all masks to `cfg.device`) makes the code slower, not faster.** Measured below.

---

## 1. Verdict on every prior-audit claim

| # | Prior claim | Verdict | Correct severity |
|---|---|---|---|
| C1 | Quest sketch upper bound sums before max | **Confirmed (math)** / **Severity wrong** | Low — dead path |
| C2 | Doc claims 100% write savings, data shows 0% | **Confirmed** / **incomplete** | High |
| C3 | "Disconnected core architecture" | **Largely incorrect** | Low |
| C4 | `initial_bifurcation` capacity invariant | **Confirmed** / unreachable via CLI | Medium (latent) |
| C5 | `baselines.py` H2O/SnapKV evict sinks | **Confirmed** / dead code | Low |
| C6 | `baselines.py` `H2O.reset_prompt` drops to 20 tokens | **Confirmed** / dead code | Low |
| W1 | Unequal attended budgets behind headline | **Confirmed**, and understated | **Critical** |
| W2 | Host–device sync in `evict_and_demote` | **Half wrong; fix is harmful** | Medium |
| W3 | `deep_demote_cost` charged for dropped tokens | **Confirmed** | Low (self-penalising) |
| W4 | Hand-picked STT-RAM parameters | **Confirmed but already disclosed** | Low / Medium |
| W5 | Quest baseline unimplemented | **Confirmed, motive misread** | Medium |
| W6 | `datasets` missing from `requirements.txt` | **Confirmed** | Low |
| S2 | `leakage_mw_per_mb` defined, never used | **Confirmed** | Low |
| S3 | `assert len(recent) >= 1` is vacuous | **Confirmed** | Low |
| Step 4 | "Move masks to `cfg.device` to reduce latency" | **Wrong — measurably slower** | — |

---

## 2. Claims the prior audit got right

### 2.1 C1 — Quest sketch bound violates the upper-bound property ✅ (math), ❌ (severity)

The reduction order in `src/tiered_kv_cache.py:235-237` and `src/baselines.py:255-257` is genuinely wrong. The code computes `max_h(Σ_d q·min, Σ_d q·max)` where Quest requires `Σ_d max(q·min, q·max)`.

Reproduced over 2000 random mixed-sign trials (D=8, page=16):

| quantity | value |
|---|---|
| trials where the code's value is **below** the true page maximum | 1217 / 2000 (60.9%) |
| worst case: true max dot product | 12.815 |
| worst case: correct Quest bound | 15.458 |
| worst case: **code's bound** | **0.787** |

So it is not an upper bound at all — it is closer to a random projection.

**But the severity is wrong.** `sketch_check()` and `_page_bounds()` have **zero call sites in `experiments/longbench_eval.py`** (verified: the only textual matches are docstrings at lines 391 and 442). The `Quest` class in `src/baselines.py` is likewise never instantiated. **No published number is affected by this bug.** It affects the CPU simulator exercised by the unit tests, and would affect any future use of the sketch path.

The audit also mislabels the second location: `src/baselines.py:255` is `Quest._sketch_score()`, not `sketch_check()`.

### 2.2 C2 — Write-savings contradiction ✅ (and worse than reported)

Both statements exist verbatim:

- `TieredKV_Project_Explanation.md:139` — *"Our experiments show **100% write savings** (zero paid writes across all benchmarks)."*
- `TieredKV_Project_Explanation.md:393` — *"measures **0% savings across all 6 headline tasks**"*

The audit stopped there. Two things it missed make this worse and stranger:

**First, both numbers are wrong.** The newest run, `results/headline_fair.json` (18:34, the only file produced by the current code), shows write savings varying by task:

| task | write_savings_pct |
|---|---|
| multifieldqa_en | 31.7 |
| hotpotqa | 0.1 |
| triviaqa | 14.0 |
| qasper | 0.0 |
| narrativeqa | 0.0 |
| gov_report | 13.6 |

**Second, the "0%" figure rests on broken statistics.** In `results/headline_equal_vram.json`, TieredKV's `cache_stats` report `decode_steps: 3` for hotpotqa and `decode_steps: 25` for multifieldqa — for tasks evaluated over **25 samples** (confirmed in `results/full6/run_gpu0.log`). Those are single-sample totals. This is exactly the cross-sample accumulator bug that the current `TieredKVPolicy.__init__` docstring says was found and fixed; the fix (`_accum`) is **not in `git HEAD`** and postdates the file. The 0% claim was computed from whichever sample happened to run last.

### 2.3 C4 — Capacity invariant ✅ (real, but unreachable through the CLI)

Reproduced with `sink_size=4, window_size=16, sram_capacity=16`:

```
after bifurcation: 20 VRAM tokens (capacity 16)
after 1 step:      20 VRAM tokens  -> overflow persists, no eviction possible
```

`initial_bifurcation` unions `pinned` into `sram_idx` without re-checking capacity, and `_hard_protected_mask()` then protects all 20, so `victims` is permanently empty.

**However, it cannot be triggered through `longbench_eval.py`'s own budget derivation.** With `recent = max(16, budget // 8)` and `tiered_vram = budget // 3`:

| `--budget` | sink+window | default `tiered_vram` | overflow? |
|---|---|---|---|
| 128 | 20 | 42 | no |
| 512 | 68 | 170 | no |
| 1024 | 132 | 341 | no |
| 1024 + `--tiered-vram-budget 1024` (headline) | 132 | 1024 | no |

It is reachable only by combining `--recent-size` with a small `--tiered-vram-budget`, or by using `TieredKVCache` directly. Real bug, latent severity. Every unit-test config keeps `sink+window ≤ sram_capacity`, so the suite cannot catch it.

### 2.4 C5 / C6 — `src/baselines.py` defects ✅ (real), ❌ (severity)

Both reproduced exactly:

```
H2O(budget=16, sink_size=4, window_size=16)
  after reset_prompt(N=50): 20 tokens
  after 3 steps:            16 tokens, positions [37..52]
  sinks 0-3 present? []          <- all four sinks wiped

H2O(budget=1024, sink_size=4, window_size=16), prompt N=4000
  kept 20 tokens          <- 1004 tokens of budget discarded
  SnapKV, same config:  kept 1024   (correct)
```

Both confirmed. The sink-eviction bug is conditional: with `budget=64 > sink+window=20`, sinks survive 60 steps intact.

**The severity is wrong for a reason the audit never checked.** `src/baselines.py` is **dead code**:

```
$ grep -rn "from baselines\|import baselines" experiments/ tests/
  NOT IMPORTED ANYWHERE
```

It is not imported by any experiment, any test, or any other module. Every baseline number in every result file comes from `H2OPolicy` / `SnapKVPolicy` / `StreamingLLMPolicy` in `longbench_eval.py`, which are separate implementations without either defect (`H2OPolicy.__call__` protects `[:start_size]` and `[seq_len-recent_size:]` and, at `budget=1024` with 132 protected, always has enough unprotected candidates).

So C5 and C6 are publication hazards — a reviewer opening `src/baselines.py` will reasonably assume it is what ran — not result-invalidating defects.

### 2.5 W1 — Unequal attended budgets ✅ (correct, and understated)

Every per-task number the audit quotes from `headline_fair.json` checks out. And the README **states the disparity itself** at line 93: *"at equal-VRAM, TieredKV attends 3072 tokens vs. H2O's 1024."*

Averages across the six tasks:

| run | Full | StreamingLLM | H2O | SnapKV | TieredKV |
|---|---|---|---|---|---|
| `headline_equal_vram.json` (README table) | 40.16 | 31.84 | 33.42 | 33.65 | **38.91** |
| `headline_fair.json` (equal attended tokens) | 43.36 | 31.78 | 33.73 | 34.20 | **35.32** |

The margin over SnapKV collapses from **+5.26** to **+1.12**.

**Two corrections to the audit's framing.** It reports the fair run as if TieredKV loses, omitting that TieredKV still has the **highest average** there and wins decisively on hotpotqa (44.27 vs 41.14) and triviaqa (54.03 vs 49.45). And its description of the *current code* is wrong: `stt_expose_quota` now defaults to `promote_top_pages × page_size = 32`, so today's code attends 1024+32 = 1056 tokens, not 3072. The 3072 figure is true of the code that produced the README, not the code in the tree.

### 2.6 W3, W4, W6, S2, S3 ✅

- **W3** confirmed. `store_dram=False` is hardcoded at `longbench_eval.py:578`, yet `deep_demote_cost` (STT read + DRAM write) is charged at line 785. For 100,000 dropped tokens the model bills 2457.6 µs / 72,089,600 nJ where only 409.6 µs / 6,553,600 nJ is defensible. **The audit omits the direction: this inflates TieredKV's own modeled cost.** The bias is self-penalising, not favourable.
- **W4** confirmed on substance but the audit presents as a discovery something the file already discloses. `cost_model.py:10-23` contains an explicit 14-line disclaimer that the STT-RAM figures are "NOT taken from a cited paper, vendor datasheet, or a named simulator run … chosen by hand". What is **not** disclosed, and is the real point, is the ordering: Tier 2 reads at 1000 GB/s and 2 pJ/bit against Tier 1's 864 GB/s and 12 pJ/bit — the nominally *slow* victim tier reads **1.16× faster and 6× cheaper** than the fast tier. Under this model there is no bandwidth reason to promote anything.
- **W6** confirmed. `longbench_eval.py:39` imports `load_dataset`; `datasets` is absent from `requirements.txt` (which does correctly pin `rouge>=1.0.1`).
- **S2** confirmed: `leakage_mw_per_mb` appears only in `TierSpec` field declarations and the three tier literals. Never read.
- **S3** confirmed: `tests/test_tiered_cache.py:64`'s `assert len(recent) >= 1` is vacuous — the just-inserted token is always present.

---

## 3. Claims the prior audit got wrong

### 3.1 C3 — "Disconnected core architecture" ❌ Largely incorrect

The audit asserts "two disconnected implementations" and that the harness "bypassed" the design. Actual call sites in `longbench_eval.py`:

| method | called? |
|---|---|
| `initial_bifurcation()` | **yes** (line 590) |
| `resolve_promotions()` | **yes** (line 745) |
| `promote()` | **yes** (line 746) |
| `evict_and_demote()` | **yes** (line 759) — this is where the inclusive shadow cache lives |
| `sketch_check()`, `_page_bounds()` | no |
| `get_peek_kv()`, `compute_attention()`, `step()` | no |

Three of the four novelties — bifurcation, promotion with hysteresis gating, and the inclusive shadow cache — run through `src/tiered_kv_cache.py` exactly as designed. What is bypassed is the Quest sketch, and the code documents why at length (`longbench_eval.py:700-713`): promotion is driven by the model's **real measured attention weights** instead of a K-as-Q sketch upper bound, which is strictly better than the approximation it replaces. `compute_attention()` is unused because the real Mistral-7B computes attention. `get_peek_kv()` is unused because peek visibility was implemented, validated, and deliberately rolled back over HF's shared causal mask (documented at lines 446-456).

The audit did not read these docstrings. It is right that the README's four-phase table (lines 31-40) misdescribes what runs — Phase 1 "Sketch" and Phase 3 "attention over the VRAM working set only" are both false — but that is a **documentation** defect, not a disconnected architecture.

**One narrow point inside C3 is valid and worth keeping:** `peek_hits` really is used only to increment a counter (`longbench_eval.py:747-750`). The reported `peeked` figure (139,232 on multifieldqa in the fair run) counts *promotion candidates that were denied by the hysteresis gate* — not tokens attended in place, which is what `metrics.py:79-82` says it means. No cost is charged for them, so nothing is inflated except the narrative.

### 3.2 W2 and Step 4 — device placement ❌ The proposed fix is harmful

The CPU-tensor allocations are real. Confirmed in `evict_and_demote`:

```
CPU | grace_t = torch.tensor(self.sram_grace, dtype=torch.bool)
CPU | already = torch.zeros(len(self.sram_pos), dtype=torch.bool)
CPU | shadow_idx0 = torch.where(torch.tensor(self.stt_shadow, dtype=torch.bool))[0].tolist()
CPU | keep_mask = torch.ones(len(self.sram_pos), dtype=torch.bool)
CPU | shadow_t = torch.tensor(self.stt_shadow, dtype=torch.bool)
CPU | priority = torch.tensor([...])
CPU | keep_mask = torch.ones(len(self.stt_pos), dtype=torch.bool)
```

But the audit's diagnosis and its fix are both wrong. Measured on the L40S at the headline shape (`H=8, D=128, VRAM=1024, STT=2048`):

| operation | CPU | GPU |
|---|---|---|
| `torch.tensor(list[2048 bool])` | 0.1004 ms | **0.1111 ms** |
| indexing a CUDA tensor with a bool mask | **0.0076 ms** | 0.0236 ms |

Creating these tensors on the GPU is **slower**, and indexing a CUDA tensor with a *CPU* mask is **3× faster** than with a GPU mask, because at these sizes kernel-launch overhead dominates the transfer. Applying the audit's Step 4 verbatim would regress `evict_and_demote`.

The overhead is real and large — it is just somewhere else:

```
evict_and_demote: 2.552 ms/call  ->  81.7 ms per decode token across 32 layers
```

That is roughly **3× a full Mistral-7B decode forward pass** on this GPU, spent entirely on cache bookkeeping. Profiling attributes it to Python-list-shaped state, not to PCIe traffic: four `torch.tensor(python_list)` constructions per call (~0.3 ms total) plus six O(n) Python list rebuilds of `stt_pos` / `stt_shadow` / `stt_was_vram_resident` (~0.4 ms total), with the remainder in inline Python and `dict(zip(sram_pos, sram_cum.tolist()))` — a forced device-to-host sync of 1024 floats, every call.

**The correct fix** is to hold `stt_shadow`, `sram_grace`, and `stt_was_vram_resident` as persistent `torch.bool` tensors that are indexed and updated in place, rather than Python lists rebuilt into tensors every step. Device choice is secondary.

The audit is also internally inconsistent here: W2 blames this code for TieredKV's GPU wall-clock regression while C3 simultaneously argues that `longbench_eval.py` does not use this module.

### 3.3 W5 — Quest stub ❌ Motive misread

`QuestPolicy.__init__` raising `NotImplementedError` is not an oversight. The docstring (lines 380-393) states that the class previously returned `past_kv` unchanged, so `--methods quest` silently produced Full-KV's numbers relabelled as "Quest". The raise was added deliberately to prevent that. This is the repository behaving *well*.

The real gap is narrower: Quest is listed in the README's "Techniques Integrated" table (line 50) as a component of the design, and the sketch scoring that would implement it is the one path with a confirmed math bug (§2.1).

---

## 4. Findings the prior audit missed

### 4.1 🔴 The README's headline results are not reproducible with the current code

`stt_expose_quota` — the parameter that bounds TieredKV's attended set — exists **8 times in the working tree and 0 times in `git HEAD`**. File times:

| artifact | modified |
|---|---|
| `results/sweep_stt/*` (README ablation) | 13:14 – 13:28 |
| `results/headline_equal_vram.json` | 13:23 |
| `README.md` | 13:29 |
| **`experiments/longbench_eval.py`** | **18:31** |
| `results/headline_fair.json` | 18:34 |

Corroborating evidence: **no result file except `headline_fair.json` contains a `tiered_stt_expose` key**, because earlier code did not write one.

Consequence: the README's Step 2 reproduction command (lines 199-205) runs today with `stt_expose_quota = 32`, producing the `headline_fair` numbers (**avg 35.32**), not the README's table (**avg 38.91**). Anyone following the README will fail to reproduce it and will not be told why.

### 4.2 🔴 `max_ctx = 8192` is hardcoded and contradicts the reference protocol

`generate_with_policy(..., max_ctx=8192)` at `longbench_eval.py:891`. `evaluate_task` calls it without passing `max_ctx`, and no CLI flag exposes it. The value cannot be changed without editing source.

The vendored reference protocol disagrees. `sota/snapkv/experiments/LongBench/config/model2maxlen.json`:

```
"mistral-7B-instruct-v0.2": 31500
```

Two consequences:

1. **Absolute scores are not comparable to published LongBench / SnapKV / H2O numbers**, despite the prompts, `maxgen` values, and metric functions being faithfully reproduced from that same reference.
2. **It structurally shrinks the effect the paper is trying to demonstrate.** At 8k context with a 1024-token budget the compression ratio is 8:1; the reference protocol's 31.5k would be ~31:1. Eviction hurts far more at 31:1, and a victim cache has far more to recover. Truncating to 8k systematically works against TieredKV's own thesis — and also caps how much prompt content exists to bifurcate into STT-RAM in the first place.

This is the most consequential hardcoded value in the repository, and the prior audit's "Absence of Hardcoding" row does not mention it.

### 4.3 🔴 Run-to-run variance exceeds the headline effect size

`results/sweep_stt/stt2048.json` and `results/headline_equal_vram.json` use the **identical** configuration (vram=1024, stt=2048, budget=1024, frac=1.0). They disagree:

| task | sweep (3-task run) | headline (6-task run) | Δ |
|---|---|---|---|
| multifieldqa_en | 59.60 | 63.57 | 3.97 |
| hotpotqa | 50.64 | 46.64 | 4.00 |
| qasper | 12.50 | 19.79 | **7.29** |

The spread comes from sample selection alone (25 samples/task, 10 for gov_report, out of LongBench's 150–200). **A ±7 F1 swing on a fixed configuration is larger than the entire +5.26 F1 margin the paper claims over SnapKV.** No result file, figure, or document reports a confidence interval, a standard error, or a seed.

### 4.4 🔴 The bifurcation-fraction ablation compares incomparable runs

`results/full6/` is presented as a single sweep. It is not:

| file | `sttram_bifurcation_frac` | samples/task | gov_report samples | config keys written |
|---|---|---|---|---|
| `baselines_and_tkv_frac1.0.json` | 1.0 | **25** | **10** | no `tiered_vram_budget` |
| `tkv_frac0.75.json` | 0.75 | **15** | **8** | has `tiered_vram_budget` |
| `tkv_frac0.5.json` | 0.5 | **15** | **8** | has `tiered_vram_budget` |
| `tkv_frac0.25.json` | 0.25 | **15** | **8** | has `tiered_vram_budget` |

Sample counts from `results/full6/run_gpu0.log` and `run_gpu1.log`. The differing config keys prove the `frac=1.0` point was produced by a **different code revision** from the other three.

`main()`'s `config_fingerprint` guards `model`, `budget`, both tiered budgets, `tiered_stt_expose`, `sttram_bifurcation_frac`, `sink`, and `recent` — but **not `max_samples` and not `tasks`**. Resuming a checkpoint after changing `--max-samples` silently blends scores computed over different numbers of samples, with no warning.

### 4.5 🔴 The README's write-savings/accuracy claim is contradicted by the repo's own data

`README.md:25` states that reserving STT headroom *"restores substantial write savings … and, in that same pilot, **improved** accuracy rather than costing it."*

The repository's own 6-task data says the opposite:

| `frac` | avg F1 | avg write savings |
|---|---|---|
| 1.00 | **28.48** | 0.0% |
| 0.75 | 26.90 | 78.5% |
| 0.50 | 27.01 | 79.1% |
| 0.25 | 25.80 | 79.7% |

Accuracy **falls** by 1.5–2.7 F1. The same holds on the three-task subset (44.50 → 40.49). The claim is unsupported by any file in the repository. (Note that §4.4 means this comparison is itself confounded, so the honest statement is that the trade-off is *unmeasured*, not that headroom helps.)

### 4.6 🟠 The headline table splices two runs from two code revisions

All 24 baseline cells in `results/headline_equal_vram.json` match `results/full6/baselines_and_tkv_frac1.0.json` exactly. All six TieredKV cells match `results/attendable_stt_validation/equal_vram.json` exactly. The two source files were written ~9 hours apart by code revisions that wrote different config keys. The README presents the merged table as one dated run.

Worth noting: in the *baseline* run's own TieredKV column (equal-total-budget, vram=341/stt=683), TieredKV averaged **28.48** — below StreamingLLM (31.84), H2O (33.42), and SnapKV (33.65). The equal-VRAM configuration is what turns that into a win.

### 4.7 🟠 The "equal-VRAM" comparison uses ~6× the GPU memory

`TieredConfig.dtype` defaults to `torch.float32` and `_bifurcate` casts with `k[0].to(torch.float32)`, while the model and every baseline hold KV in bfloat16. Combined with the STT-RAM tier being physically resident in VRAM:

| configuration | actual GPU KV memory |
|---|---|
| H2O / SnapKV / StreamingLLM @ 1024 tokens, bf16 | 134.2 MB |
| TieredKV VRAM tier, 1024 tokens, **fp32** | 268.4 MB |
| TieredKV STT tier, 2048 tokens, **fp32**, also in VRAM | 536.9 MB |
| **TieredKV total resident** | **805.3 MB (6.0×)** |
| plus the bf16 copy `_visible_tuple` rebuilds every step | +138.4 MB |

The simulated nature of the STT tier is disclosed; the fp32 storage (a silent 2×) is not, and `CostModel(bytes_per_elem=2)` prices the hierarchy as if it were fp16. "Equal-VRAM" is a claim about a hypothetical system, not about measured occupancy.

### 4.8 🟠 Zero test coverage of the code that produced every published number

All 10 tests import only `src/tiered_kv_cache.py`. Untested entirely: `TieredKVPolicy`, `_visible_tuple`, `_decode_step`, `update_scores`, `H2OPolicy`, `SnapKVPolicy`, `StreamingLLMPolicy`, `_ModeledCostMixin`, `generate_with_policy`, and the checkpoint fingerprint logic.

This is where the gap bites: **both known-fixed bugs** — the cross-sample stats accumulator and the unbounded STT exposure — lived in `longbench_eval.py`, and both were found by inspection after publishing numbers, not by a test. `src/baselines.py` (§2.4) has no coverage either, which is how C5 and C6 survived.

### 4.9 🟡 H2O and SnapKV baselines use keys as a proxy for queries

`H2OPolicy._initialize_scores` (line 202) and `SnapKVPolicy._compress_prompt` (line 351) both do `q_win = k[:, :, seq_len - w:, :]` — the observation window's **keys** standing in for its queries. Both papers use real queries. The pooling is otherwise faithful (avg_pool1d, kernel 5, padding 2 — matching `sota/snapkv/.../snapkv_utils.py:57`).

The limitation is documented in `TieredKVPolicy`'s class docstring and applied uniformly at prefill. But TieredKV's *decode-time* promotion uses the model's **real** attention weights (`update_scores`), while the baselines' decode-time eviction also uses real attention — so the asymmetry is confined to prompt compression, where it hits SnapKV hardest, since prompt compression *is* SnapKV's entire contribution. This is a fairness issue running in the opposite direction from W1, and the README's results discussion does not mention it.

### 4.10 🟡 Smaller items

- **`_visible_tuple` has a single point of failure.** `n_stt_uni = min(stt_expose_quota, min over layers of len(live))`. One layer with zero live non-shadow STT tokens silently zeroes the exposed slow tier for **all 32 layers**, collapsing TieredKV to a VRAM-only policy with no error and no log line.
- **README ablation "monotonic" claim overstated.** Only the 3-task average is monotonic (35.77 → 38.79 → 40.91 → 44.98). Per task, qasper goes 13.06 → 15.85 → **12.50** → 23.62 and hotpotqa 41.09 → 40.76 → 50.64 → 50.64.
- **`sweep_stt_budget.sh` does not reproduce the README ablation.** The script defaults to 5 tasks and `SAMPLES=15`; the results contain 3 tasks and the README says 10 samples.
- **README self-contradiction on test count.** Lines 131 and 241 say "7 unit tests"; line 174 says "10 tests". Actual: 10.
- **`snapkv_importance` edge attenuation.** `avg_pool1d` defaults to `count_include_pad=True`, so importance at positions 0–1 and N-2–N-1 is divided by 5 while fewer real values contribute. Harmless here — all four positions are pinned as sinks or window — but it matches the reference implementation's behaviour, so no change is needed.

---

## 5. What the repository does genuinely well

Worth stating plainly, because the prior audit's tone undersells it and a reviewer will notice these:

1. **Explicit RoPE position tracking** (`longbench_eval.py:946`, `true_pos = prompt_len + step`). Every sparse policy trims the cache, so HF's default `arange + get_seq_length()` would assign false relative distances. Passing true document positions for every policy alike is correct and is the kind of thing most reimplementations get wrong.
2. **Faithful LongBench scoring.** Prompts, `maxgen` values, ROUGE-L on raw text via the real `rouge` package, ambiguity-splitting classification, and fractional retrieval/count credit all match `sota/snapkv/experiments/LongBench/` exactly. The first-line trim for `trec`/`triviaqa`/`samsum`/`lsht` matches the reference too.
3. **Checkpoint config fingerprinting** (lines 1251-1280) that *raises* rather than silently blending configs. The gap in §4.4 is that `max_samples` is not in the fingerprint — the mechanism itself is sound.
4. **The `QuestPolicy` raise and the `cost_model.py` hardware disclaimer** are both deliberate honesty guards, written to stop a wrong number reaching a table. The prior audit counted both against the repo.
5. **Extensive negative-result documentation.** Rolled-back peek visibility, the median-vs-min sketch threshold, the age-normalised eviction score, and the `sttram_bifurcation_frac` trade-off are each documented with the measurement that motivated them. This is unusually good engineering hygiene.

---

## 6. Recommended priority order

Ordered by effect on whether the work survives review, which is close to the inverse of the prior audit's ordering.

| # | Action | Why |
|---|---|---|
| 1 | **Re-run the headline and every ablation with the current code**, then rewrite the README's tables from those files. | §4.1 — nothing in the README is currently reproducible. Everything else is downstream of this. |
| 2 | **Decide and state which comparison is the paper's claim**: equal-VRAM (3072 resident tokens) or equal-attended (1056). Report both tables side by side. | §2.5 — the +5.26 margin is +1.12 under equal attention. A reviewer will find this. |
| 3 | **Raise `max_ctx` to 31500** (or expose it as a CLI flag and report the value used). | §4.2 — 8k truncation both breaks comparability and suppresses the effect being claimed. |
| 4 | **Report variance.** Increase to ≥50 samples/task, fix a seed, and publish per-task standard errors. | §4.3 — measurement noise currently exceeds the claimed effect. |
| 5 | **Delete the "100% write savings" sentence** (`Explanation.md:139`) and the "improved accuracy" clause (`README.md:25`); replace both with the measured per-task table from a single clean run. | §2.2, §4.5 — the most direct integrity exposure. |
| 6 | **Add `max_samples` and `tasks` to `config_fingerprint`**; re-run the `frac` sweep at one fixed sample count. | §4.4 — the ablation is currently uninterpretable. |
| 7 | **Fix the README's Phase 1/Phase 3 architecture table** to describe attention-driven promotion and two-tier attention. | §3.1 — the documented pipeline is not the one that runs. |
| 8 | **Convert `stt_shadow`, `sram_grace`, `stt_was_vram_resident` to persistent `torch.bool` tensors**; keep the masks on CPU. | §3.2 — 81.7 ms/token of bookkeeping. Do **not** apply the prior audit's Step 4. |
| 9 | **Either delete `src/baselines.py` or fix and wire it in.** If kept, fix `reset_prompt` to honour `budget` and guard `argmin` against an all-`inf` candidate vector. | §2.4 — dead code that reads as the experiment. |
| 10 | **Fix the sketch bound** to `torch.maximum(q*mn, q*mx).sum(-1).sum(-1)` in both files, and add a property test asserting `bound ≥ max(q·k)` over random vectors. | §2.1 — required before the sketch path is used or a real Quest baseline is built. |
| 11 | **Guard `initial_bifurcation`** against `sink_size + window_size > sram_capacity`. | §2.3 |
| 12 | **Add `datasets` to `requirements.txt`**; reconcile "7 tests" → 10; charge only the STT read when `store_dram=False`; either compute leakage or delete the field; rename or drop the `peeked` metric. | §2.6, §3.1, §4.10 |
| 13 | **Add integration tests for `longbench_eval.py`** on a tiny random model: budget invariants per policy, sink survival, `_visible_tuple` length uniformity, and cross-sample stats accumulation. | §4.8 — the only untested code is the code that produces results. |

---

## 7. One-line summary of the prior audit

Technically competent at spotting local defects in files it read, and unreliable on severity because it did not check which code actually runs, did not read the docstrings explaining the deviations it flagged, did not compare result files against the code that produced them, and proposed at least one fix that is measurably harmful.
