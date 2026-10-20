# Current research checkpoint — 20 October 2026

Read this file, then TASK_STATE.json and REPRODUCING.md. This is the current checkpoint. Older dated plans, logs, checkpoints and archived prose are historical evidence, not active instructions. No SOTA result or publication readiness is established; paper writing remains paused until the research gates finish.

## Recover the working project

1. Follow the clean setup in README.md/REPRODUCING.md. Run `python scripts/restore_evidence.py` to restore oversized raw records and `python scripts/bootstrap_references.py` for pinned core selectors.
2. Run the documented core tests, simulator smoke and separate GQA-selector tests. Optional `--all` fetches IceCache/RocketKV sources; it does not build or validate them.
3. Inspect actual processes/GPU users. The recorded 16 October supervisors have terminated. There are no known running research jobs at this checkpoint. Do not relaunch old queues using stale PIDs or old status prose.

## Completed evidence

- `results/full_evaluation_20261015/verified_results.json`: 10,318 predictions over 1,474 original six-task records/seven configurations, independently verified. The recorded-unconsumed five-task Tiered hot992 versus legacy faithful Snap1024 contrast was −1.36577 points, 95% CI[−3.73347,0.33088]. Preserve this negative outcome. Resources differ.
- `results/gqa_baseline_20261016/verified_results.json`: the additional official current GQA SnapKV run is COMPLETE, 1,474 requests independently rescored. Task macro 39.52297; existing Tiered hot1024 macro 38.80885. No superiority. This descriptive population is consumed; it cannot tune a new variant.
- Existing native-versus-legacySnap timing failed advancement (~53 versus 26 ms/token). Internal controller acceleration is not SOTA acceleration. Preserve all failed/successful results and their source/configuration epochs.
- Whole-request write accounting independently checked for 1,474 records each of hot992/hot1024. Figure 3 uses hot992: 40.6793% pooled avoidance including initial retained cold placement, 26.0881% equal-task macro, 16.7657% five-QA pooled. Its 91.3811% decode-only pooled figure is not whole-request savings. No physical STT endurance inference.
- Figures 1/2/3/4/6/7/11/12/14 use verified matching runs. Figures 8/15 remain pending/deferred. Main adapted controls cannot support SOTA. Architecture SVG/PDF/PNG is current; preserve original plot questions/style.

## Next actions, in order

1. Diagnose the stopped IceCache GPU preflight. `results/audit_20261016_checkpoint/icecache_kernel_tiny_model_v2.log` shows a 2D RMSnorm harness input where the official kernel expects 3D. Preserve that failure; register a corrected harness/new output path, then validate kernels, original model path, proposed layout correction through real offload, K/V scatter/retrieval and pretrained 7B before comparison. Offload validation did not execute after its prerequisite failed. CPU projection/index/layout proofs alone are insufficient. Index RNG reproducibility remains open.
2. Diagnose RocketKV receipt check failure (`rocketkv_runtime_resume_status.json` under proper_baselines_20261016): the CPU check assumes direct_url archive metadata always has `hashes`; installed metadata uses another valid form. Preserve the failure, fix receipt parsing without changing algorithms, then validate FlashAttention, real RoPE/model patch and common-runtime controls. The source is pinned63637c8eb4b06b5eb83cd687723b0eeb19f48f65. Fetching/importing is not model correctness.
3. Freeze explicit GPU/host resource envelopes and valid official configurations before fair quality/latency/storage comparisons. Same initial token count is not equal whole-memory fairness. Charge index/work buffers, retained cold copies, growing decode and TTFT. Use fastest correct supported baseline backends and identical model/input/generation/scoring. Read `results/proper_baselines_20261016/comparison_contract.json` and `paper_baseline_scope_review.json`.
4. Seek distinct, evidence-based improvements on development data only. Grouped-retention, query-combination and exposure-prior screens failed their registered gates; do not rerun/enlarge them for favorable outcomes. Current-query page approximation was rejected as duplicate expansion.800 cross-task rows/635 target groups remain reserved and unscored; verify their manifest before any prospective use. Never tune on the consumed 1,474 final records.
5. Complete the broader audit, missing ablations and matching original-style figures. Resolve novelty: FlexiCache already retains host copies and offloads stable-head pages once, so write-once backup alone is not a new contribution. A global SOTA claim cannot follow from comparing only named SnapKV/IceCache controls. Only after these gates, draft/refine the four-page IEEE paper.

## Reproducibility rules

Use new output directories whenever source/runtime/configuration changes. Keep raw failures and successes immutable. Historical result runners intentionally preserve original server paths/hashes; do not edit them to pretend they are portable or resume them under mismatched source. Use the documented portable CLI for new runs and preregister each paper experiment. STT simulation, GPU-functional quality, analytical costs and CPU-offload timing are separate scopes.

Update this checkpoint and TASK_STATE.json at meaningful milestones. Do not create additional competing checklists. Historical narrative/old paper sources are in docs/archive; prior detailed state remains as immutable evidence in results/publication_20261020/before_TASK_STATE.json.
