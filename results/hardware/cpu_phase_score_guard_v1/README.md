# CPU phase controller with score fallback guard

498 tests pass. A shared batched scorer now rejects multi-query fast paths before mutation, routing them through unchanged ordinary scoring fallback. Existing single-query arithmetic is unchanged. Real2/32-token GPU appends preserve all7 logits/KV/metadata/slot boundaries and return to native single-token mode. See ../cpu_phase_append_fallback_v1/verified_fallback.json; both failed attempts are preserved (score crash, then Transformers-owned lazy hook initialization in the tiny harness).

Current-source4K12 and fresh16K50 match frozen behavior/native transfers. Active GPU+pinned+CPU array bound433.333MiB; prefill peak unchanged. Quiet3ABBA300 measured forwards/arm:58.757 to54.120ms median,7.892% faster/4.637ms saved. P95 improves8.873%; TTFT+1.373%, all guards pass. No fallback. See verified_correctness.json, verified_resource.json, verified_cost.json.

Actual distinct KV roots272.25MiB are constant; archived faithful SnapKV487–512MiB grows0.5MiB/step. Width1024 differs on49/50steps. No fixed integer Snap prompt capacity can match this allocation vector. See verified_geometry.json. The separate433.333MiB scoped peak includes rounded pinned blocks and metadata, not merely KV backing capacity. Figures include both the ordinary-control speed result and the unmatched baseline geometry, with limits.

Pre-guard boundary runs cover exclusive24832 promotions, floor-above-capacity and50 write-aware fallback steps, with1152 native CPU cold reclaims. Their snapshots stay archived with measured source; current single-query paths are refreshed here. The multi-query scorer fallback keeps the ordinary scorer's existing single-query semantics; it does not add query aggregation.

All owned jobs terminal. Research-only; no default adoption or full P0/P1/P2 completion. Equal-resource/comparable-width comparison design, independent quality, physical controller measurements, broader generality/fair external baselines and final supported paper remain open. Unsupported day estimate withdrawn.
