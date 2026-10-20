# Flow diagram review — 15 October 2026

Read-only review of both existing JPGs against the current executable model policy and standalone simulator. No diagram, implementation, experiment, or GPU state was changed. STT is intentionally simulated; physical STT access is not a completion gate.

## Assets and source availability

- `figures/tieredkv_flow_diagram.jpg`: 768×1376, colored rounded panels, sans serif text, orange/teal/red initialization cards and blue/green/orange/red decode panels.
- `figures/tieredkv_flow_diagram_academic.jpg`: 768×1376, grayscale rounded panels, serif text, dashed separators, explicit backup decision diamond.
- Both first appear in commit `20b3143`; its tree contains these raster files but no corresponding editable source. Searches of repository text, repository filenames and the surrounding thesis/home paths found no diagram generator, SVG, Draw.io, PPTX, Mermaid or Excalidraw source for these assets. This is a searched-source result, not proof no source exists elsewhere.
- `TieredKV_Project_Explanation.md:150` embeds the academic JPG and already warns that the DRAM tier was retired. Its subsequent query-sketch claim at approximately line156 is also misleading for the model policy and must be qualified if the figure's explanatory prose is updated.
- `explanation.md` has Mermaid diagrams, but they are not sources for either JPG.

## Factual corrections needed in both JPGs

| Existing diagram claim | Executable evidence | Minimal correction |
|---|---|---|
| “3-Tier KV Cache Memory Hierarchy”; “Tier 3: DRAM / Drop” | `src/tiered_kv_cache.py:187` and `:889` route into two retained tiers; excess live rows are permanently discarded. No third payload tier. | Title “TieredKV: Two-Tier KV Cache”; keep the third card in its existing place but label it “Dropped Tokens / Not a storage tier / Permanently discarded”. |
| Cold bandwidth “1TB/s read / 250GB/s write” | `src/cost_model.py:53` sets nominal assumed cold50/12.5 GB/s; these are analytical sensitivity inputs, not measured GPU-visible throughput. | Prefer remove numerical bandwidth from the architecture drawing; use “Simulated STT cost model”. If retained, write “Assumed50/12.5 GB/s”; never attribute it to pinned CPU transfer speed. |
| Initial cold tier contains “Shadow backup tokens” | `src/tiered_kv_cache.py:265` initializes all `stt_shadow` flags false. `:606` marks a row shadow on inclusive promotion. | “Live victim rows / Shadows after promotion”. |
| Fixed128 hot /256 cold tokens as architecture constants | `TieredConfig`, `TieredKVPolicy.__init__`, CLI factory expose capacities; current reusable-root timing runner uses1024/2048. | “Hot budget: H tokens”; “Cold budget: C tokens”. Do not replace stale examples with one experiment's values as universal constants. |
| Hot initialization only sinks+window (academic version) | `src/tiered_kv_cache.py:238` also selects highest-importance unpinned rows. | “Sinks + recent + top scores”. |
| “1. Sketch Check” as the model's first per-token phase | `experiments/longbench_eval.py:885` selects visible live cold rows by last access; `:854` promotes measured-attention candidates after the forward at `:1255`. No `sketch_check` call in this model decode. `experiments/cold_boundary.py:74` preserves grouped timestamp ranking. | Panel1 “Prepare Visible KV”: hot rows plus quota-limited live cold rows; recency selection; skip shadows; same exposed count across layers. |
| Promotion before the current token's model attention | `generate_with_policy` runs model forward, updates scores, then invokes policy at `:1255–1264`. `_decode_step` promotes using that feedback. | Reorder panel contents within the unchanged four panel slots: visible KV → model attention/feedback → promotion → eviction/demotion. |
| “Exact attention over VRAM set ONLY” | `_visible_tuple` concatenates hot and selected cold; model appends current token. `src/tiered_kv_cache.py:977` standalone attention also includes deferred peek rows. | “Attention: hot + selected live cold + current token”. Shadow duplicates excluded. |
| No promotion hysteresis / live floor | `src/tiered_kv_cache.py:413`, `:480`, `:550`; real-model `_promotion_plan` calls index hysteresis. | Promotion panel says “Attention-ranked candidates; budget + re-entry gate”; optionally “Preserve live-cold floor”. |
| “Evict lowest cumulative-attention unprotected token” | `src/tiered_kv_cache.py:687` uses cumulative/age; optional write-aware subtraction at `:689`; hard sinks/recent protected, grace relaxed if capacity requires. | “Demote lowest age-normalized score; protect sinks/recent”. Add small “optional backup preference” only if space allows. |
| “Reclaim LRU shadow” | `src/tiered_kv_cache.py:880` prioritizes highest cumulative-score hot twin; orphan shadow first. Live rows use last-access LRU. | “Reclaim shadows first (highest-score hot twin); then LRU live rows”. |
| “Zero write cost” / “free” shadow reactivation | Existing backup avoids a new payload write; metadata/controller work remains, and this is not zero whole-loop latency. | “Reuse shadow: no KV payload write”. Keep YES/NO decision and its existing geometric style. |
| “Else LRU real victim → DRAM” | No third-tier store; live-floor handling may discard demoted tail to stay bounded (`src/tiered_kv_cache.py:903`). | “Then LRU live rows / Permanent drop; enforce capacity and live floor”. |

## Recommended final panel text, preserving the existing style

Preserve canvas size, rounded rectangles, connectors, panel positions, typeface family, color palette/grayscale and decision diamond. Only replace labels/content. The four decode slots should show the executed model flow:

1. **Prepare Visible KV**
   - Hot rows + quota-limited live cold rows
   - Recency selects cold exposure; shadows excluded
   - Same exposed count across layers
   - Native option: gather cold KV from pinned CPU pool
2. **Model Attention & Feedback**
   - Attend hot + selected cold + current token
   - Produce next-token logits; collect real attention
   - Update hot scores and cold access timestamps
3. **Promote Attention-Ranked Cold Rows**
   - Apply promotion budget, re-entry gate and live floor
   - Inclusive: retain cold shadow backup
   - Exclusive ablation: remove promoted cold copy
4. **Evict & Demote**
   - Admit current token KV to the hot tier
   - Overflow: lowest age-normalized score, protect sinks/recent
   - Backup? YES: reactivate, no KV payload write
   - NO: write KV and append live cold victim
   - Cold overflow: reclaim shadows, then LRU live rows
   - Discard overflow permanently; preserve live floor where feasible

Footer/caption: “Model decode policy. STT costs are analytical; native CPU storage is an optional backend. Standalone sketch/peek simulation is separate.” The initial visible set is prepared after prefill; panel1 represents the prepared set at each iteration, while `_decode_step` prepares the subsequent set after panel4. Output logits arise in panel2; the existing bottom output box can remain as delivery after bookkeeping, but label it “Emit next token” rather than implying the cache controller computes logits.

Initialization subtitle: “Observation-window importance (key proxy by default; actual-query option)”. The current opt-in actual-query prefill uses captured rotated queries and GQA causal scoring (`experiments/longbench_eval.py:708`); this does not turn decode exposure into all-cold current-query search.

If the user instead wants the **standalone simulator** diagram, label that scope explicitly and preserve sketch → promotion/peek → attention → eviction ordering. Add read-only peek rows, current-token attention before eviction, and the ideal-persistent-bound accounting limitation. Do not merge simulator and model paths into one unlabeled sequence.

## Performance mechanisms supported by current code

These are operation-level observations, not new benchmark results or predicted savings.

1. **GPU/CPU decision synchronization remains in every decode.** `experiments/batched_score_policy.py:39` converts folded cold scores to a Python list; `experiments/probe_batched_eviction_selection.py:48` ranks GPU scores and copies index/count decisions to CPU; `experiments/cold_boundary.py:74` performs GPU timestamp argsort then `.cpu().numpy()` for CPU physical address planning. These are concrete cross-device dependencies. Measure their current contribution before proposing further changes; do not add nested profiler times as independent savings.
2. **Promotions and next-visible cold reads are separate payload phases.** Native preparation prefetches promotions through `NativeControllerPolicy.native_prepare` and shared-pool reads; `cold_boundary.py:81` then flushes paid writes, waits on the manager event, and transfers selected cold rows. Exact immutable KV shadows avoid repeated paid writes but do not eliminate reads, address construction or stream synchronization. Any reuse must preserve which rows were read, their slot versions and cross-stream lifetime.
3. **Model attention feedback requires eager weights on the current harness.** `longbench_eval.py:1255` requests `output_attentions=True`; the main loader explicitly uses eager decode (`:1629`). The real attention feedback is consumed for eviction and promotion, so switching attention backends without retaining those statistics changes policy behavior. A future optimization must first preserve the received score vectors numerically and include extraction cost.
4. **CPU orchestration remains across all layers even with native payload kernels.** `cpu_phase_policy.py:164` performs geometry/ownership validation, native CPU transitions and per-layer preparation; `native_controller_policy.py:110` materializes promotion frames; `bulk_payload_policy.py:75` builds grouped executor descriptors. `cpu_property` converts versioned position arrays into tuples (`cpu_phase_policy.py:30`). Current native kernels remove many copies/launches, not all host work.
5. **Whole-context prefill is still a resource cost.** Ordinary bifurcation at `longbench_eval.py:705` casts full keys/values toF32 before scoring and storing reduced-precision selected rows; actual-query scoring chunks observation queries but spans the prompt. Exact pooled decode storage does not itself make prefill bounded. Bounded-prefill is a separate approximate mode requiring separate quality acceptance.
6. **Root allocation was already addressed, not a fresh optimization opportunity.** `experiments/reusable_exact_root.py:37` reuses one exact pinned root with serial same-context leases. It counts retained idle storage and releases only on final close. Existing allocation diagnoses and reusable-root screens are historical evidence; do not promise their gains again for a new candidate. The current Snap projection-reuse change likewise already removes duplicate prefill Q/K projections, default-off pending its comparison protocol.

Historical profile `results/hardware/cpu_phase_policy_profile_v1/attribution.json` includes cold initialization/build imports and resource observers; its cumulative times overlap and its source epoch predates current corrections. It supports where to inspect, not the current runtime ranking. Historical reusable-root verification claims an internal controller speed/resource pass; this review did not rerun or independently reverify that full raw experiment. A fair faithful-Snap full-loop comparison still requires the planned source-matched, symmetric resource/timing protocol.

## Validation and limits

Visually inspected both complete raster images. Read current continuity files, traced prefill/model attention/promotion/eviction/native visible composition, inspected allocator/query-score and profile code, searched source/history for editable diagram files. No model, GPU benchmark or test suite was run for this figure review. Original JPG hashes and code citation hashes are recorded in `flow_review.json`. No physical hardware claim or independent quality claim is made.
