# TieredKV (KV_Cache_Optimization) — Independent Harsh Audit
**Date:** 2026-09-14 | **Scope:** full repo at `KV_Cache_Optimization/` | **Method:** read every `src/` + `experiments/longbench_eval.py` end-to-end, parsed all `results/*.json`, checked `figures/`, `tests/`, `sota/`, `README.md`, `TieredKV_Project_Explanation.md`, `AUDIT_VERIFICATION.md`. Re-ran JSON cross-checks and one test-collection probe. No files edited.
**Verdict up front:** technically competent prototype, **not production-ready, not top-tier publishable as-is, and not beating SOTA under any fair definition.** The headline (+5.26 over SnapKV) is an attended-tokens artifact. Under equal attended tokens it is +1.12 → +0.33 (noise). Under equal total memory it loses at every budget. The repo's own internal audit (`AUDIT_VERIFICATION.md`) is unusually honest and already concedes the biggest problems — this report verifies those concessions by execution, adds new bugs it missed, and judges conference readiness.

---

## 0. Bottom line (read this first)

| Question | Answer |
|---|---|
| Is it correctly implemented in parts? | Yes — a minority of the code is genuinely good (§1). |
| Is it technically sound end-to-end? | **No.** At least 2 result-invalidating harness bugs remain (§2), plus variance > effect (§4). |
| Is it production-ready? | **No.** ~81.7 ms/token bookkeeping across 32 layers (~3× a Mistral-7B forward pass), 3–5.5× wall-clock slower than H2O, O(n) Python-list rebuilds per step, no batching/serving path, simulated NVM with all tensors in one VRAM allocation. |
| Does it beat SOTA? | **Only when given ~3× the tokens.** Equal-VRAM headline: 38.91 vs 33.65 SnapKV (+5.26) at 3072 vs 1024 attended. Equal-attended (`headline_fair.json`): 35.32 vs 34.20 (+1.12). Clean rerun at `max_ctx=31500` (`rerun_31500/`): +0.33, with per-task losses. Equal-total (`budget_sweep/`): **dead last at 256/512/1024/2048.** |
| Is the shadow-cache novelty demonstrated? | **No.** 0.0% savings on all 6 headline tasks; 0–31.7% per-task on the fair run (0% on 3/6 tasks); 78–79% only with headroom at −1.5 to −2.7 F1 cost, from an uninterpretable 25-vs-15-sample sweep. |
| Are figures top-tier ready? | **No.** Data-driven but stale (visualize irreproducible pre-quota JSON), no error bars, radar charts, JPG flow diagram, hardcoded axes, duplicated generators, wrong-config latency/PPL figures. |
| Publishable where? | Workshop/tech-report after fixes. **NeurIPS/ICML/ISCA/MICRO: reject** in current state (§7). |

The single most consequential fact, already stated in `AUDIT_VERIFICATION.md §0` and verified here: **every published number in `README.md` was produced by a version of `longbench_eval.py` that no longer exists.** Repro commands today yield ~35.3, not 38.91.

---

## 1. What is correctly implemented (credit where due)

1. **Quest bound fix is real.** `src/tiered_kv_cache.py:272-305` computes per-dim `maximum(q*min, q*max)` then sums — the correct Quest upper bound. The old sum-before-max form fails ~60.9% of mixed-sign trials. `src/baselines.py:290-307` matches. Property test `tests/test_sketch_bound.py` is mutation-verified.
2. **Bifurcation / promotion / demote path is actually wired** (contrary to a prior "disconnected architecture" claim). `experiments/longbench_eval.py:590` calls `initial_bifurcation()`, `:745-746` `resolve_promotions()`+`promote()`, `:759`/`870` `evict_and_demote()`. Only the sketch, `compute_attention()`, and peek-visibility are bypassed — and the bypass is documented (`longbench_eval.py:700-713`, `README.md:42-48`) with a sound reason (real measured attention > K-as-Q sketch).
3. **Capacity guard, sink protection, grace, hysteresis.** `tiered_kv_cache.py:166-181` raises on `sink+window > sram_capacity`; `590-620` prefers ordinary victims; `466-474`/`752` one-shot grace; median (not min) threshold `325-338` avoids grace-token churn; shadow-reclaim-highest-twin-first `685-722` is correct.
4. **RoPE position tracking.** `longbench_eval.py:946` (`true_pos = prompt_len + step`) passes true document positions for every policy. Most sparse-cache reimplementations get this wrong. Genuinely good.
5. **Faithful LongBench scoring.** Prompts, `maxgen`, ROUGE-L via real `rouge`, normalization, fractional retrieval credit match `sota/snapkv/experiments/LongBench/`. Verified by inspection.
6. **Fingerprinting + honesty guards.** `longbench_eval.py:1410-1431` raises on config mismatch (now covers model/budget/tiers/expose/frac/sink/recent/max_samples/tasks/max_ctx/seed/dtype); `QuestPolicy` raising `NotImplementedError` instead of silently returning Full-KV is correct behavior; `cost_model.py:10-23` explicitly discloses hand-picked STT numbers; `tiered_kv_cache.py:42-80` explicitly discloses the fp32 6× footprint. The candor is a strength — but disclosed flaws still count against acceptance.
7. **Negative-result documentation.** Rolled-back peek visibility, median-vs-min, age-normalized eviction, bifurcation trade-off are each documented with measurements. Unusually good hygiene.

---

## 2. What is incorrectly implemented, inefficient, or flawed

### CRITICAL (result-invalidating)

**C1. `TieredKVPolicy.reset()` double-counts starvation/exposure — NEW, missed by internal audit.**
`longbench_eval.py:592-603`: `_live_totals()` (`574-590`) already includes `stt_starved_steps/stt_exposed_total/stt_exposure_samples`; lines 594-596 fold all of them into `_accum`, then lines 600-602 add the same three **again**. Starved totals are ~2× inflated and means double-weight prior samples. Every `cache_stats` starvation figure in every result file is wrong. The existing test (`test_stats_accumulate…:166-188`) only checks `decode_steps` (folds once) so it passes despite the bug.

**C2. `deep_demote_cost` charged while `store_dram=False` — still in the harness.**
`longbench_eval.py:649` sets `store_dram=False`; line 896 still calls `cache.cm.deep_demote_cost(dropped…)`. `cost_model.py:123-146` documents this overbills ~6× latency / ~11× energy vs `drop_cost()`. `AUDIT_VERIFICATION.md` marks this "Done" — it is fixed in `src/tiered_kv_cache.py:822-826` but **not** on the harness path that produced all LongBench numbers. Self-penalizing (inflates TieredKV's own cost) but proves modeled-cost columns are wrong in both directions.

**C3. Headline is attended-tokens, not hierarchy.** Verified by parsing JSONs:
- `headline_equal_vram.json`: no `tiered_stt_expose` key → unbounded, ~1024+2048=3072 attended vs 1024 for baselines. Avg 38.91 vs SnapKV 33.65 (+5.26).
- `headline_fair.json` (vram=992+expose 32=1024): 35.32 vs 34.20 (+1.12). TieredKV **loses qasper 8.62 vs 12.44** outright.
- `rerun_31500/` (same 25 samples, `max_ctx=31500`, current code): equal-VRAM (expose 2048) 43.03 vs 35.65 (+7.38); equal-attended (expose 32) 35.98 vs 35.65 (**+0.33**), losing hotpotqa and narrativeqa outright.
- `budget_sweep/` (only equal-*total* comparison, vram=budget//3): TieredKV **last at every budget** — 256: 28.40 vs 28.66; 512: 29.39 vs 31.47; 1024: 31.07 vs 34.84 (spot-checked multifieldqa 43.75 vs 53.86–58.03, hotpotqa 27.87 vs 35.19–37.63); 2048: 33.05 vs 38.83.
No figure/table presents all three definitions side by side. The paper's claim depends entirely on which denominator is hidden.

**C4. Output length uncontrolled, confounds every F1/ROUGE comparison.**
`generate_with_policy:1076-1113` runs to `max_gen-1` or EOS per policy. On identical samples (`headline_fair.json`): multifieldqa TieredKV 269 steps vs 63 for every baseline (4.3×); triviaqa 188 vs 31 (6×); qasper 421 vs 15–18 (**28×**); narrativeqa 113 vs 3–13 (37×); gov_report 3025 vs 258–409 (11×). QA-F1 penalizes verbosity via precision; ROUGE rewards length via recall. Length is never reported, normalized, or controlled. The STT "ablation" moves qasper output length 127 vs 13 vs 18 while scores move 12.50 vs 15.85 vs 23.62 — it is a length artifact, not a scaling law.

**C5. Variance exceeds effect; no CIs; provenance missing.**
Same config (vram=1024/stt=2048) disagrees across files: qasper 12.50 (`sweep_stt/stt2048.json`) vs 19.79 (`headline_equal_vram.json`), Δ**7.29 > +5.26 headline margin, > +1.12/+0.33 fair margins**. Within-run logs show ±22-point swings on 5-sample increments. `--seed` is a placebo for scores (first-N slice `1175-1177`, model is argmax eval; seeds at `1319-1320` don't touch selection). Published JSONs (`headline_*`, `sweep_stt/*`) carry **no** `max_samples/max_ctx/seed`. `plot_*.py` plot means only. No result is statistically distinguishable from noise.

**C6. Spliced + incomparable runs presented as single experiments.**
- All 24 baseline cells in `headline_equal_vram.json` byte-match `full6/baselines_and_tkv_frac1.0.json`; all 6 TieredKV cells match `attendable_stt_validation/equal_vram.json` (~9h apart, different config keys). Presented as one dated run.
- `full6/` "bifurcation sweep": frac=1.0 at 25/task (10 gov_report) by one code revision (no `tiered_*_budget` keys) vs frac 0.75/0.50/0.25 at 15/task (8 gov_report) by another (has keys; `tasks` even says `["gov_report"]` while results contain 6 tasks). Uninterpretable.
- `max_ctx` truncation alone moves Full scores triviaqa 66.08→74.64, gov_report 18.63→33.60 (8192 vs 31500) — more than the claimed margin.

### HIGH (major, not immediately fatal)

- **Default fp32 billed as fp16; 6× resident memory.** `TieredConfig.dtype=float32` default (`:65`), `resolved_kv_dtype=float32` (`:84-85`), `_bifurcate` casts `k[0].to(torch.float32)` (`longbench_eval.py:658`), while model/baselines hold bf16 and `CostModel(bytes_per_elem=2)` prices fp16. Headline "1024-token equal-VRAM": TieredKV VRAM tier 268.4 MB vs baseline 134.2 MB + 536.9 MB "slow tier" also in VRAM + 138.4 MB rebuilt `_visible_tuple` copy = **~805 MB (6.0×)**. Opt-in `--tiered-kv-dtype` doesn't retroactively fix published numbers.
- **`sttram_bifurcation_frac=1.0` default defeats the victim cache by construction.** `tiered_kv_cache.py:60,190-192` fills STT to 100% at step 0 on any long doc. The file itself (`:42-51`) admits `write_savings_pct=0.0%` end-to-end "regardless of promotion rate or eviction scoring." The default guarantees the headline mechanism cannot fire.
- **Wall-clock 3–5.5× slower than H2O; modeled cost 11× worse.** `README.md:128` concedes the former (`evict_and_demote` 2.55→2.12 ms/call → ~81.7 ms/token across 32 layers, ~3× a forward pass, from Python-list rebuilds + `dict(zip(sram_pos,…))` host sync per call). `headline_fair.json` modeled totals show TieredKV 0.70s/53.9J vs SnapKV 0.06s/5.2J territory per task — the paper's own efficiency metric refutes an efficiency win.
- **STT "slow tier" reads faster/cheaper than VRAM.** `cost_model.py:52-63`: STT 1000 GB/s / 2 pJ/bit vs VRAM 864 GB/s / 12 pJ/bit. Under this model there is no bandwidth reason to promote anything. Hand-picked and disclosed (`:10-23`) — still unpublishable as a hardware result without cited NVSim/CACTI/Destiny numbers or a sensitivity framing.
- **MACs tracked but never monetized; threshold matmul hidden.** `tiered_kv_cache.py:836-842` latency = sketch-read + promote + peek-read + VRAM-read + demote only. `attn_macs` (`attention.py:35`) never converts to time/energy; `lat_attention_us:827` always prices full `sram_tokens`, not the attended subset; the per-step full `q @ sram_k.T` threshold matmul (`:325-326`) over all of VRAM is uncounted while `sketch_macs:347` counts only `2*H*pages*D`.
- **Latency/PPL figures use different TieredKV configs on neutered paths.** `latency_profile.py:56-69` measures vram=budget//3 (equal-total, not headline 1024/2048), with `output_attentions=False:118-121` and no `update_scores:126-127` → promotion never fires; synthetic `"quick brown fox"*3000` prompt. Result (~24.9 ms ITL ≈ H2O) contradicts README's "3–5.5× slower" because it doesn't measure the eval loop. `ppl_eval.py:162-165` runs TieredKV at sram=1024/stt=1024/expose=512 (1536 attended vs 1024, undisclosed in `fig4_ppl_vs_context`); `ppl_results.json:config.methods=["snapkv"]` but results contain 5 methods (resume-overwrite provenance break).
- **K-as-Q proxy for baselines, real attention for TieredKV.** `longbench_eval.py:231,388,660` (`q_win = k[…-w:]`) score H2O init, SnapKV compress, and bifurcation with keys-as-queries (no RoPE, no causal query semantics). TieredKV decode promotion uses real attention (`update_scores:912-964`). The asymmetry hits SnapKV hardest (prompt compression *is* SnapKV) and is undisclosed in results discussion.
- **`_visible_tuple` single point of failure.** `:735` `n_stt_uni = min(quota, min live)` — one fully-shadowed layer zeroes slow-tier exposure for all 32 layers with no error. Counted/warned now, still silently degrades to VRAM-only (rerun shows 14 starved steps on hotpotqa).
- **`peek_attn > 0` vacuous.** `longbench_eval.py:961-962` / `tiered_kv_cache.py:798`: softmax outputs are always >0, so every "peeked" token refreshes `stt_last_access`. LRU protection meaningless. `peeked` counts hysteresis-deferred candidates, not in-place attention (`metrics.py:79-82` says otherwise).
- **Tests don't reproduce.** `python -m pytest tests/ --collect-only` in the shipped venv errors on `test_longbench_policies.py` (`ModuleNotFoundError: No module named 'rouge'`) despite `rouge>=1.0.1` in requirements; 29 tests collect, 1 errors. "42 passing" is environment-dependent. Coverage still misses output-length/EOS behavior, K-as-Q asymmetry, middle-split seam, `deep_demote` vs `drop_cost` branch, and C1 above.

### MEDIUM/LOW (fix before submission)

- Middle-split truncation (`:1050-1052` `[:half]+[-half:]`) assigns contiguous `arange(prompt_len)` (`:1060`), creating false adjacency at the seam for all methods — relatively fair internally, breaks comparability to published LongBench.
- Stale docstring `:1027` says `max_ctx (<=8192)` while default/flag is 31500 (`:1008,:1244`).
- `snapkv_importance` edge deflation (`avg_pool1d` default `count_include_pad=True`) + head collapse (`attention.py:60-67` sums to single `(N,)`; real SnapKV is per-head).
- Prefill-seed scale mismatch: `cum=importance` (~H×W mass) vs decode increments ~1/step; promoted tokens restart at 0 (`evict_and_demote:648-672` discards cum/age) — structural bias against re-entry, interacts with hysteresis=2 ("structural minimum, not tuned" `:353-391`).
- Page-granularity over-gating (`:413-414` `any(was_resident)` + `min(streak)` blocks whole page for one churner); transient VRAM overshoot to capacity+promoted+1 mid-step (`promote:457-458`, `evict:544-545`) while `step:813-814` records post-eviction occupancy only; DRAM-spill `torch.cat` per token (`:731-734`) O(k·n); `leakage_energy_nj` (`cost_model.py:148-165`) never called; `promotions_deferred` never written (`tiered_kv_cache.py:771,781`); inconsistent default windows (`baselines.py:56` StreamingLLM 64 vs H2O/SnapKV 16; Quest top-4 vs TieredKV top-2); `from attention import…` requires `PYTHONPATH=src` (`:17-19`).

---

## 3. Hardcoded values, shortcuts, "cheating" — direct answers

1. **`max_ctx=8192` was hardcoded** (`generate_with_policy(..., max_ctx=8192):891`, unexposed, contradicting reference `sota/snapkv/.../model2maxlen.json` Mistral-7B 31500). Now default 31500 + `--max-ctx`, but all headline/ablation JSONs predate the fix. Truncation shrank the compression ratio from ~31:1 to ~8:1 — against the paper's own thesis.
2. **Magic numbers everywhere, almost none tuned:** sink=4, window=16, sram=64, sttram=128, page=16, promote_top=2, pool=5, hysteresis=2, clamp_min=1, grace=1, median threshold, `bytes_per_elem=2`. Only hysteresis and the bound justify their constants.
3. **Simulated tiers, one allocation.** `tiered_kv_cache.py:97-98,123-124,139-140` allocate all tiers as `torch.empty(..., device=cfg.device)` in VRAM. No pinning, no H2D/D2H, no NVM model. Every latency/energy number is analytical (`cost_model.py:52-70`). Legitimate for a simulator — not for a "measured STT-RAM" claim.
4. **Bandwidth/energy inversion (slow > fast).** See §2 HIGH. The nominally slow victim tier reads 1.16× faster and 6× cheaper than the fast tier. Either the numbers are wrong or promotion is pointless.
5. **fp32 residency priced as fp16.** See §2 HIGH. Silent 2× on top of the 3× token-count difference.
6. **Unbounded exposure behind the headline.** Pre-quota code attended the whole live slow tier (~3072 vs 1024). README now admits it (`:74-79`); current default quota 32 (`:537-540`) means repro commands produce the fair numbers, not the headline.
7. **No real Quest baseline; sketch dead on harness.** `QuestPolicy` raises by design (honest), but README lists Quest as "integrated" (`:50,58`), and no Quest numbers exist anywhere. Comparing "vs SOTA" while omitting the one baseline sharing your sketch lineage is a gap reviewers will flag.
8. **Vendored `sota/` unused.** `sota/{h2o,quest,snapkv,streamingllm}/` are reference clones, never imported. All baseline numbers come from harness reimplementations with the K-as-Q proxy above — not upstream code.
9. **Sweep scripts don't reproduce sweeps.** `budget_sweep.sh` says 6 tasks/25 samples; JSONs are 6 tasks/20 samples. `sweep_stt_budget.sh` now pins 3 tasks/25/seed-0; JSONs have no seed/expose keys and multifieldqa migration counters are bit-identical across stt1024/2048/3072 (demoted 30593/promoted 28577) — capacity doesn't bind there, so the "sweep" varies a non-binding parameter.
10. **No `cheating` in the sense of fabricated numbers** — JSONs are internally consistent and the internal audit's candor is real. The problem is subtler and worse for review: **fairness bugs (C3–C4), noise > signal (C5), and stale/unreproducible tables (C6)** that collectively make the SOTA-beating claim unsupportable, plus self-penalizing cost bugs (C2) that undermine the systems story from the other side.

---

## 4. Experiments, SOTA comparison, figures

- **Main comparison:** invalid as published. Must report equal-VRAM, equal-total, and equal-attended side by side with CIs, at one revision, one sample count (≥50/task), fixed seed, `max_ctx=31500`, bf16 residency. Current state: win iff given 3× tokens; tie at equal attention; loss at equal memory.
- **Ablations:** STT-budget "monotonic" claim is mean-only (35.77→38.79→40.91→44.98); per-task qasper 13.06→15.85→**12.50**→23.62, hotpotqa 41.09→40.76→50.64→50.64. Window-size ablation scripted (`sweep_window_size.sh`) but never run. Bifurcation-fraction sweep uninterpretable (§2 C6) and shows headroom *costs* 1.5–2.7 F1 — the opposite of the withdrawn "improved accuracy" claim.
- **Figures:** `plot_*.py` read JSONs (not hardcoded — verified averages match exactly), but visualize stale pre-quota data; two parallel generators diverge (lexicographic vs numeric sort — `table_stt_ablation.tex` orders 1024/2048/3072/**512** last); axes hardcoded (`plot_all_figures.py:597-598` pareto `ylim(30,48) xlim(0,135)` clips 31500-context runs); captions assert configs matching only specific JSONs; `fig6_migration_profile` labels "Peeked (in-place)" for events the harness says never happen (`:849-858`); radar charts discouraged at top venues; flow diagram is JPG not vector; no error bars anywhere despite noise > effect. **Not conference-grade.**
- **Tables:** `results/tables/` duplicates schemes (`accuracy_*.tex` + `table1/2/3/4_*.tex`); lack std/significance/maxgen/prompt details.
- **SOTA coverage:** H2O/StreamingLLM/SnapKV reimplemented (with proxy); Quest absent; no InfLLM/ShadowKV/PQCache/InfiniGen/RetrievalAttention/SAGE-KV/FIER/LMCache/FlexKV (2024–26 retrieval literature) — fatal for a 2026 submission claiming a retrieval-style hierarchy.

---

## 5. Novelty assessment (2026 bar)

| Claimed contribution | Verdict |
|---|---|
| 3-tier GPU→NVM→DRAM hierarchy | **Recombination.** FlexGen (ICML'23) did GPU→CPU→disk with optimizer + 4-bit KV; InfLLM (NeurIPS'24), ShadowKV, PQCache, RetroInfer, InfiniGen (OSDI'24), LMCache/FlexKV/KVDrive all do tiered retrieval. NVM-for-KV is not new. |
| Demote/promote victim cache | **Relabelled retrieval.** Once STT is attendable, this is exactly the literature's "KV retrieval" (retain + select subset/step: Quest/InfLLM/ShadowKV) vs "KV dropping" (H2O/SnapKV/StreamingLLM). Classic Jouppi victim-cache applied to KV. |
| Inclusive shadow / zero-write re-demote via KV immutability | **Only candidate micro-novelty — unproven.** Inclusive caches + KV immutability are both standard (every prefix-cache relies on immutability). The STT-asymmetry application is a reasonable arch micro-optimization, but default-config result is 0%, fair-run 0–31.7%, headroom trade costs accuracy. No paper result yet. |
| Attention-driven promotion + hysteresis | **Minor heuristic, negative-novelty direction.** Full exact attention over VRAM+STT to decide promotion defeats Quest's purpose (avoid scoring everything). Streak=2 untuned, no theory (cf. H2O submodular guarantee). |
| Quest sketch | Borrowed, dead on harness, was buggy (now fixed). Cannot be claimed. |
| SnapKV/H2O/StreamingLLM/FlexGen integration | Acknowledged recombination — honest, but hurts the ML-novelty bar. |

Net: competent systems integration; algorithmic novelty ≈ one cache-policy tweak with no demonstrated win. Workshop/tech-report level as-is.

---

## 6. Are the figures relevant for a top-tier conference?

No. Even if the science were fixed: no uncertainty visualization, wrong/stale configs, radar + JPG, missing Pareto (accuracy vs total-bytes vs attended-tokens vs modeled energy), no budget sweep at equal-total, no multi-model/seed curves, no wall-clock-vs-modeled reconciliation, no endurance/area/leakage analysis for an NVM paper. A reviewer will also notice TieredKV exceeding Full-KV on 3/6 tasks (63.57 vs 60.95 etc.) — possible via different EOS/length behavior, but without length control it reads as a harness artifact, not a finding.

---

## 7. What it would take (priority order)

1. **Re-run everything** at one revision, ≥50 samples/task, fixed seed, `max_ctx=31500`, bf16 residency. Rewrite all tables from those files. Nothing else matters until this is done.
2. **Report all three fairness definitions side by side** (equal-VRAM / equal-total / equal-attended) with per-task SE/CIs. Pick the headline *after* seeing them.
3. **Control output length** (fixed-length scoring or length-stratified analysis; report decode_steps per method).
4. **Fix C1 (double-count) + C2 (deep_demote branch)** in harness; add regression tests for both.
5. **Fix cost model:** cite real STT-MRAM (or reframe as sensitivity study); fix bandwidth inversion; price fp32 honestly or switch to bf16; reconcile modeled vs wall-clock; include leakage/endurance/area.
6. **Decide the shadow-cache story:** demonstrate savings without accuracy loss or drop the claim. Fix `frac` sweep at one sample count first.
7. **Run real baselines:** wire upstream `sota/` or current releases; add Quest + at least InfLLM/ShadowKV; survey 2024–26 retrieval work.
8. **Reduce bookkeeping** (persistent bool tensors already done; next: vectorize compaction, eliminate per-step host syncs) and add a batched-serving throughput story for arch venues.
9. **Figures:** error bars, Pareto, equal-total budget curves, vector flow diagram, provenance-stamped generation (config hash in caption), delete stale/duplicate outputs.
10. **NeurIPS/ICML additionally:** algorithmic insight beyond integration (theory or non-trivial policy) + full LongBench + ≥2 model families + component-isolating ablations. **ISCA/MICRO additionally:** NVSim/CACTI/Destiny or gem5 modeling, NVM parameter sensitivity, DRAM/PCM/ReRAM comparison, batched throughput.

---

## 8. One-line summary

A candid, well-documented prototype whose headline SOTA win evaporates under equal attention or equal memory, whose only novel mechanism measures 0% at default settings, and whose evaluation pipeline has more noise than signal — **fix the harness, re-run at one config with statistics, and resubmit as a workshop paper with a narrower claim; do not submit to a top-tier venue in this state.**
