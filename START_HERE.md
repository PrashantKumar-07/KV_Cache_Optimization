# Current research checkpoint — 20 October 2026 (task list updated 20 November 2026)

Read this file, then TASK_STATE.json and REPRODUCING.md. This is the current checkpoint. Older dated plans, logs, checkpoints and archived prose are historical evidence, not active instructions. No SOTA result or publication readiness is established; paper writing remains paused until the research gates finish.

## Moving to a new server (added 18 November 2026)

If this clone is on a new machine, set it up with the steps below before anything else. Storage location does not change results: no code path stores KV cache on disk (tiers live in GPU/host memory), so SSD vs HDD only affects model load time. Do not change code, configs, seeds or pinned versions to "fit" the new server.

Reference runtime of the original server (match it): Linux x86_64, Python 3.10.12, torch 2.5.1+cu124 (CUDA 12.4), transformers 5.15.0, NVIDIA L40S 48 GB. Check the new GPU/driver with `nvidia-smi`; a different GPU type needs its own validation and must be recorded, not mixed silently with L40S timings.

Storage layout (Server 2: fast shared NVMe `/home`, ~257 GB free; large HDD `/data`, ~6.9 TB free):
- NVMe (`~`): the git clone and the Python venv (~10 GB total; small, read constantly).
- HDD (`/data/<user>/kvcache_store`): model weights (~28 GB), Hugging Face cache, LongBench data, and optional official-baseline toolchains (IceCache/RocketKV, ~14 GB, only when rerunning those baselines). Linked into the repo with symlinks so all relative paths (`models/...`, `data/longbench`) keep working unchanged.

```bash
# 1. Clone to fast storage
cd ~ && git clone https://github.com/PrashantKumar-07/KV_Cache_Optimization.git
cd ~/KV_Cache_Optimization

# 2. Large-storage dirs on HDD + symlinks (models/, data/ are gitignored)
STORE=/data/$USER/kvcache_store
mkdir -p $STORE/models $STORE/data $STORE/hf_cache
ln -s $STORE/models models
ln -s $STORE/data data
echo "export HF_HOME=$STORE/hf_cache" >> ~/.bashrc && export HF_HOME=$STORE/hf_cache

# 3. CUDA environment on fast storage (ignored by git as .venv-cuda/)
python3.10 -m venv .venv-cuda && source .venv-cuda/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check

# 4. Restore evidence and pinned references
python scripts/restore_evidence.py          # must end "Verified 7 original research records"
python scripts/bootstrap_references.py      # add --all only when working on IceCache/RocketKV

# 5. Model and data (pinned revision / verified hashes)
python download_model.py --output models/mistral-7b-instruct
python scripts/download_longbench.py --output data/longbench
```

Alternative to step 5: copy the already-downloaded files from the old server with `rsync -a --info=progress2` into `$STORE/models/mistral-7b-instruct` and `$STORE/hf_cache`, then still run `scripts/download_longbench.py` (it verifies the six JSONL hashes; use `--archive` to reuse a copied archive offline).

Verify before any experiment (all must pass):
```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python scripts/smoke.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=experiments python -m pytest -q results/improvement_design_20261016/test_gqa_snap_adapter.py
mkdir -p results/local_smoke_server2
CUDA_VISIBLE_DEVICES=0 python experiments/longbench_eval.py \
  --model models/mistral-7b-instruct --data-dir data/longbench \
  --tasks multifieldqa_en --methods full tieredkv --budget 1024 \
  --max-samples 1 --prefill-last-logits --tiered-kv-dtype bfloat16 \
  --json results/local_smoke_server2/comparison.json
```

Notes for the new server:
- Upstream method repos (H2O, SnapKV, StreamingLLM, Quest) under `sota/` are gitignored and were not migrated; the pinned references come from `bootstrap_references.py`. Re-clone upstream into `sota/` only if a task explicitly needs them.
- Old historical `results/**/run.py` scripts contain original-server paths on purpose (provenance). Do not edit or resume them; use the portable CLI with a new output directory.
- Write all new outputs to new, server-labelled directories (e.g. `..._server2`) so new hardware runs are never mixed with recorded L40S evidence. Timing/latency comparisons are only valid within one server.
- The `/home` NVMe is shared with other users and nearly full; keep large outputs and checkpoints on `/data`.

## Recover the working project

1. Follow the clean setup in README.md/REPRODUCING.md. Run `python scripts/restore_evidence.py` to restore oversized raw records and `python scripts/bootstrap_references.py` for pinned core selectors.
2. Run the documented core tests, simulator smoke and separate GQA-selector tests. Optional `--all` fetches IceCache/RocketKV sources; it does not build or validate them.
3. Inspect actual processes/GPU users. The recorded 16 October supervisors have terminated. There are no known running research jobs at this checkpoint. Do not relaunch old queues using stale PIDs or old status prose.

## Completed evidence

- `results/full_evaluation_20261015/verified_results.json`: 10,318 predictions over 1,474 original six-task records/seven configurations, independently verified. The recorded-unconsumed five-task Tiered hot992 versus legacy faithful Snap1024 contrast was −1.36577 points, 95% CI[−3.73347,0.33088]. Preserve this negative outcome. Resources differ.
- `results/gqa_baseline_20261016/verified_results.json`: the additional official current GQA SnapKV run is COMPLETE, 1,474 requests independently rescored. Task macro 39.52297; existing Tiered hot1024 macro 38.80885. No superiority. This descriptive population is consumed; it cannot tune a new variant.
- Existing native-versus-legacySnap timing failed advancement (~53 versus 26 ms/token). Internal controller acceleration is not SOTA acceleration. Preserve all failed/successful results and their source/configuration epochs.
- Whole-request write accounting independently checked for 1,474 records each of hot992/hot1024. Figure 3 uses hot992: 40.6793% pooled avoidance including initial retained cold placement, 26.0881% equal-task macro, 16.7657% five-QA pooled. Its 91.3811% decode-only pooled figure is not whole-request savings. No physical STT endurance inference.
- Final figures are renumbered 1–9 (18 Nov 2026; old→new map in figures_final/README.md) and all use verified matching runs. Unverified old Figures 8/15 and the old diagrams/figure archive were removed (still in git history). Architecture: figures/tieredkv_architecture_v2.svg. Current figure renderer/verifiers (write the new names; byte-identical to October): results/figures_20261118_renumbered/. Legacy plotting scripts were replaced by experiments/figure_style.py; originals stay in the frozen source.tar.gz. Main adapted controls cannot support SOTA. Architecture SVG/PDF/PNG is current; preserve original plot questions/style.

## Next actions, in order (updated 20 November 2026)

This is the single task list. It merges the earlier gates with the paper plan agreed after the November presentation review. Do the setup above first. Paper writing stays paused until steps 1–7 are done.

### 1. Sharpen the novelty claim
"Write once" alone is not new: FlexiCache (MLSys 2026, §3.3) offloads each stable-head KV page once and keeps a full host mirror. The defensible claim is managing a **bounded, write-expensive non-volatile tier**: write accounting, shadow reclamation on overflow, and quality under a write budget. Compare against FlexiCache directly (design table at least; experiment if its code is available). Step 2 may add a second novelty.

### 2. Better choice of which cold rows join attention (exposure selection) — candidate second novelty
Problem: each step only 32 cold rows join attention, chosen by recency (`_visible_cold_indices` in `experiments/longbench_eval.py`, sort by `stt_last_access`). Rows outside the 32 get no attention, so they can never be promoted ("recency blindness"; proven in `results/improvement_design_20261016/cold_query_page_opportunity.json`). A retained fact can be effectively lost.

Already tried by earlier work — read before starting, do not repeat:
- Exact current-query ranking of all cold rows (`experiments/evaluate_query_selection.py`): **+2.92 points vs recency** on 30 consumed development rows, 95% CI [−0.29, 6.67]. It failed only because the CI crossed zero (sample too small). So there may be real headroom. Record: `results/audit_20261015_query_combination/verified_quality.json`. Do not rerun it on the same rows or enlarge it under a new name.
- Quest-style min/max page bounds: rejected (poor recall: about 6.9 of the true top 32 with 8 pages; loose bounds; raw q·k bound is not a bound on the softmax objective). Do not use min/max as the selector.
- Centroid page index: failed its screen (top-32 overlap 0.28).

Candidates to test against the recency control (pick whichever wins the pre-registered gate):
- **A. Rotating exploration slots** (cheap baseline fix): e.g. 24 slots by recency + 8 rotating through the cold tier, so every cold row is shown at least once every 2048/8 = 256 steps. No row can starve.
- **B. Score-based exposure**: rank cold rows by decayed accumulated attention instead of last-use time.
- **C. Confidence-triggered, write-free cold retrieval (proposed new novelty)**:
  - Trigger: the attention over hot rows is computed anyway. When it is confident (low entropy, little mass on sinks), keep recency. When it is uncertain (high entropy / mass dumped on sinks), give most of the 32 slots to a search of the whole cold tier. The model signals when it is missing context.
  - Search: a tiny key fingerprint per cold row (e.g. 128-bit SimHash per KV head, Hamming distance to the hashed query). About 256 KB per layer, ~8 MB total vs 256 MB cold payload.
  - Write-aware placement: the fingerprint index lives in GPU memory (write-cheap), never in the STT-RAM tier, so it adds **zero NVM writes**.
  - Keep a few rotating slots (as in A) as a no-starvation guarantee.
  - Literature check first: FLARE (confidence-triggered retrieval in RAG), MagicPIG / HashAttention (hashing keys), InfiniGen, Loki, Quest. Claim only the combination (uncertainty trigger for a bounded write-expensive KV tier + index placed to add no NVM writes) if it is not already published.

Procedure:
1. Build a **fresh development set** (not the 1,474 consumed records, not the 800 reserved rows). Register it with hashes before running.
2. Cheap diagnostic first (no new method needed): an oracle that scores all 2,048 cold rows exactly and exposes the true top 32, against recency. Log attention entropy per step and check whether high-entropy steps are where the oracle and recency disagree. If the oracle gains nothing, stop here and report it. If the entropy signal does not predict disagreement, drop the trigger and keep A/B.
3. Freeze one configuration per candidate and the acceptance gate before seeing outputs (existing rule: ≥ +1.0 task-balanced point vs recency with cluster-95% lower bound > 0, no task loss > 5 points; see `prospective_135_gate` in `cold_query_page_opportunity.json`). Count extra reads, index memory and time in the resource envelope.
4. Adopt the winner; keep all losing runs as preserved evidence.

### 3. Fair, equal-memory comparison
Today TieredKV holds more total memory (398.5 MB vs 134.2 MB per request). Compare at **matched total GPU+CPU memory** against the official SnapKV and at least one modern method (RocketKV or IceCache). Both are blocked by setup failures, not algorithm problems:
- IceCache: GPU preflight stopped on an RMSnorm harness shape mismatch (2D supplied, kernel expects 3D; `results/audit_20261016_checkpoint/icecache_kernel_tiny_model_v2.log`). Preserve the failure, register a corrected harness and a new output path, then validate kernels, offload, K/V scatter/retrieval and pretrained 7B before any comparison. Index RNG reproducibility remains open.
- RocketKV: receipt check assumes `direct_url` metadata always has `hashes` (`results/proper_baselines_20261016/rocketkv_runtime_resume_status.json`). Fix the parsing without changing the algorithm (pinned source 63637c8eb4b06b5eb83cd687723b0eeb19f48f65), then validate FlashAttention, real RoPE/model patch and common-runtime controls.
- Freeze explicit GPU/host resource envelopes and valid official configurations first. Same initial token count is not whole-memory fairness. Charge index/work buffers, retained cold copies, growing decode cache and TTFT. Read `results/proper_baselines_20261016/comparison_contract.json` and `paper_baseline_scope_review.json`.
- Fairness options to report: (a) equal attended rows (current hot992 + 32), (b) equal total storage (shrink TieredKV hot+cold to the baseline total), (c) equal peak memory for growing-cache baselines (cap SnapKV's growth at the same total).

### 4. Fix the latency
About 75–78 ms/token vs 23–34 ms for the others; reviewers will reject this. Options: a fused GPU kernel for the per-layer policy (scoring, victim choice, promote/demote), and/or a real CPU-memory cold tier with measured timing. Earlier native-controller attempt: ~53 vs 26 ms/token, failed advancement — preserve it. Timing is valid only within one server.

### 5. Make the STT-RAM numbers credible
Replace assumed speeds/energies (`src/cost_model.py`) with values cited from published device papers or a simulator such as NVSim. Run a sensitivity sweep (e.g. write energy and write latency ×0.5 to ×4) and show the conclusions hold. Keep "logical writes, not physical endurance" wording.

### 6. Missing ablations
Without shadows (every demotion pays a write), without exposure (cold tier invisible until promoted), different cold-tier sizes, different exposure quotas (8/16/32/64), and the exposure selectors from step 2. Also the optional headroom/cold-fill-fraction ablation (old unverified Figures 8/15 were removed).

### 7. Broader tests
1–2 more models (e.g. Llama-3-8B) and longer contexts (64K–128K). Use the 800 reserved, never-scored rows (`results/improvement_design_20261016/fresh_document_reserve.json`; verify its manifest first) **only once**, for a final pre-registered test after everything is frozen. Never tune on the 1,474 consumed records.

### 8. Then write the 4-page IEEE paper
Only after steps 1–7. No SOTA claim unless a fair comparison supports it.

### Small presentation fix (optional)
`docs/presentation/QUICK_SUMMARY.md` novelty 2 says "no approximation in the loop". More accurate: "every step, the 32 most recently used cold rows join attention together with all hot rows; the model's real attention on those 32 decides which move back to the GPU." Rebuild the PDF with `md2html.py` as described in `docs/presentation/README.md`.

## Reproducibility rules

Use new output directories whenever source/runtime/configuration changes. Keep raw failures and successes immutable. Historical result runners intentionally preserve original server paths/hashes; do not edit them to pretend they are portable or resume them under mismatched source. Use the documented portable CLI for new runs and preregister each paper experiment. STT simulation, GPU-functional quality, analytical costs and CPU-offload timing are separate scopes.

Update this checkpoint and TASK_STATE.json at meaningful milestones. Do not create additional competing checklists. Historical narrative/old paper sources are in docs/archive; prior detailed state remains as immutable evidence in results/publication_20261020/before_TASK_STATE.json.
