# Exact-pool/shared-feedback critical review

No blocker found. Independent focused CPU suite:28 passed, exit0, log `exact_pool_guard_focused.log`; source/log hashes are in `exact_pool_guard_review.json`. No GPU jobs or source edits.

`ExactBulkPayloadPool(BulkPayloadPool)` preserves operational constructor/read/queue_bulk/flush/enqueue and native address counters; allocation is the sole override. ExactColdBoundaryPolicy preserves decode, visibility, native prepare and reset inherited methods; ordinary policy stays on BulkPayloadPool. This corrects the incompatibleDirectSharedPool substitution identified before implementation.

The shared `_validate_score_weights` checks every nonNone layer before either update path mutates score/LRU or resets attention dictionaries. Batch!=1, ndim!=4, zero heads or zero queries reject; positive ordinary/batched fallback arithmetic remains unchanged. Focused tests verify malformed final layer leaves earlier caches and dictionaries intact and valid1/2/32/64-query behavior. This is structural feedback validation, not a guarantee of finite/arbitrarily adversarial weights or all generic models.

Allocation lifetime still requires serialized sameCUDAcontext consumers. Exact allocator finalizer waits before free and depends on final view/alias release; verify live stats return to baseline after every cleanup, public observed allocationrange, full-model payload/logit parity and identical registered resource thresholds in new immutable outputs. Constructor failure rollback is outside the inherited allocator-only change; guarded runner resource creation is required. Current cached-pool741.65MiB rejection stays historical evidence. No resource/timing pass is inferred for exact roots before measurements.
