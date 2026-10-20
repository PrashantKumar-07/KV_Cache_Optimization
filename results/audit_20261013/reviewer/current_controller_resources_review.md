# Independent current-controller resource gate

Completed read-only CPU verification, no trial reruns or source/runner/raw edits. Runnable proof: `results/audit_20261013/current_controller_resources/verify_resources.py`; successful exit0, `verification.log` and `verified_resources.json`. Verification succeeds while **registered resource advancement is rejected**.

All5913 archive/plan/live source hashes and runtime matched at2026-10-14T06:49:10Z before subsequent permitted source edits; separate `independent_source_attestation.json` preserves this checkpoint. Original runner/plan/prompt/teacher/quiet-helper hashes match. All12 model artifacts, including three large weight shards, were independently rehashed. The previously independently verified full-logit/payload prerequisite is bound by its evidence hashes. Twelve ABBA rows contain600 measured decode forwards per arm, exact repeated/cross-arm predictions matching that prerequisite, matching logical counters,25 valid quiet observations and closed hooks/leases. Final-trial native cold/CPUappend/bulk steps50 each, CPU/bulk fallback0. Cold-boundary fallback1 is expected initial-prefill visibility before CPU arena attachment; subsequent50 decode visibility calls are native, so this is not a failed decode guard.

| Registered statistic | Control | Candidate | Ratio | Guard |
|---|---:|---:|---:|---|
| Pooled medianITL(ms) |79.4957978651|50.7097374648|0.63789204998|Pass≤0.90|
| P95ITL(ms) |85.6197340414|54.7932904214|0.63996099772|Pass≤1.05|
| MedianTTFT(ms) |2273.92559592|2401.99600626|1.05632128447|Pass≤1.10|
| PeakdecodeGPUallocated(GB) |15.135259136|14.671673344|0.96937047540|Pass≤1.00|
| PeakprefillGPUallocated(GB) |18.627831296|18.627831296|1.0|Pass≤1.10|
| ScopedGPU+CPUcache(MiB) |not this guard's control metric|741.654663086|—|**Fail≤512**|

Measured median internal-controller speed gain36.210795% is useful evidence; it does not override the failed combined-memory guard or establish a faithful-SOTA result/default adoption. ABBA block median ratios0.65054/0.62028/0.65929 (exact values in proof); these are descriptive blocks, not confidence intervals.

Every candidate row has the same registered memory sum777,681,280B:

- GPU peak minus deduplicated static model storage188,208,640B (179.489746MiB).
- Observed pinned main-pool publicCUDA allocator block536,870,912B (512MiB), although requested storage276,824,064B (264MiB).
- Canonical CPU arrays1,800,832B; peak CPU streak/address arrays206,848B+65,536B.
- Registered conservative descriptor reservation50,528,512B:2×final-trial cumulative bulk/cold descriptor traffic.

Even pool+GPU alone totals691.489746MiB, so rejection does not depend on descriptor-reservation interpretation. Do not substitute requested pool bytes, drop terms, loosen threshold or repeat the same measured candidate. GPU totals include model/workspaces; formula subtracts only explicitly deduplicated constant model storage. Host probe executes after measured loops. Scoped evidence excludes hidden cached host/driver blocks, Python/native container rounding andRSS; no whole-system-memory equality is claimed. Quiet observations bracket rows rather than continuously monitoring utilization. One reused16K teacher-forced input is not natural generation, quality/generalization, production adoption or physicalSTT evidence.

## Exact-pool follow-up reuse warning

`ExactSharedPool` inherits `DirectSharedPool`; directly installing it on currentColdBoundaryPolicy removes `BulkPayloadPool.bulk_pending`, `queue_bulk`, bulkflush behavior and `ControllerPool` native address/counter handling. Current bulk `_prepare_decode_reads` dereferences those fields and invokesqueue_bulk. Minimal safe artifact-only variant subclasses **BulkPayloadPool** and overrides **onlyallocate_pool=staticmethod(exact_pinned_memory.allocate)**; policy overrides direct_pool_cls to that new pool. Existing exact allocator can be reused, but requires new source epoch/parity/resources and observed live allocator range, not assumed264MiB allocation. Exact driver's finalizer is proven for serialized single-context use, synchronizes that context before free, and releases only when all tensor/views disappear. Verify live allocation statistics return to baseline after leases/cache/pool aliases close, and preserve asynchronous manager ordering. No production concurrency safety inferred.
