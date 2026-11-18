# Current research checkpoint — 20 October 2026

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
- Final figures are renumbered 1–9 (18 Nov 2026; old→new map in figures_final/README.md) and all use verified matching runs. Unverified old Figures 8/15 and the old diagrams/figure archive were removed (still in git history). Architecture: figures/tieredkv_architecture_v2.svg. Main adapted controls cannot support SOTA. Architecture SVG/PDF/PNG is current; preserve original plot questions/style.

## Next actions, in order

1. Diagnose the stopped IceCache GPU preflight. `results/audit_20261016_checkpoint/icecache_kernel_tiny_model_v2.log` shows a 2D RMSnorm harness input where the official kernel expects 3D. Preserve that failure; register a corrected harness/new output path, then validate kernels, original model path, proposed layout correction through real offload, K/V scatter/retrieval and pretrained 7B before comparison. Offload validation did not execute after its prerequisite failed. CPU projection/index/layout proofs alone are insufficient. Index RNG reproducibility remains open.
2. Diagnose RocketKV receipt check failure (`rocketkv_runtime_resume_status.json` under proper_baselines_20261016): the CPU check assumes direct_url archive metadata always has `hashes`; installed metadata uses another valid form. Preserve the failure, fix receipt parsing without changing algorithms, then validate FlashAttention, real RoPE/model patch and common-runtime controls. The source is pinned63637c8eb4b06b5eb83cd687723b0eeb19f48f65. Fetching/importing is not model correctness.
3. Freeze explicit GPU/host resource envelopes and valid official configurations before fair quality/latency/storage comparisons. Same initial token count is not equal whole-memory fairness. Charge index/work buffers, retained cold copies, growing decode and TTFT. Use fastest correct supported baseline backends and identical model/input/generation/scoring. Read `results/proper_baselines_20261016/comparison_contract.json` and `paper_baseline_scope_review.json`.
4. Seek distinct, evidence-based improvements on development data only. Grouped-retention, query-combination and exposure-prior screens failed their registered gates; do not rerun/enlarge them for favorable outcomes. Current-query page approximation was rejected as duplicate expansion.800 cross-task rows/635 target groups remain reserved and unscored; verify their manifest before any prospective use. Never tune on the consumed 1,474 final records.
5. Complete the broader audit, missing ablations and matching original-style figures. Resolve novelty: FlexiCache already retains host copies and offloads stable-head pages once, so write-once backup alone is not a new contribution. A global SOTA claim cannot follow from comparing only named SnapKV/IceCache controls. Only after these gates, draft/refine the four-page IEEE paper.

## Reproducibility rules

Use new output directories whenever source/runtime/configuration changes. Keep raw failures and successes immutable. Historical result runners intentionally preserve original server paths/hashes; do not edit them to pretend they are portable or resume them under mismatched source. Use the documented portable CLI for new runs and preregister each paper experiment. STT simulation, GPU-functional quality, analytical costs and CPU-offload timing are separate scopes.

Update this checkpoint and TASK_STATE.json at meaningful milestones. Do not create additional competing checklists. Historical narrative/old paper sources are in docs/archive; prior detailed state remains as immutable evidence in results/publication_20261020/before_TASK_STATE.json.
