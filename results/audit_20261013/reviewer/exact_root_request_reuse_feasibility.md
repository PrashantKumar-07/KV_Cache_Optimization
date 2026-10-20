# Exact-root sequential-request reuse: narrow feasibility

Read-only source/evidence review, no new timing/GPU work or source edits. Current exact gate rejectsTTFT; this report does not reinterpret its guard.

## Evidence-ranked cause

Measured exact candidate prefillforward median1886.2431ms matches ordinary1886.8237ms, while TTFT−forward median675.0919 versus387.3018ms. Cached-pool epoch candidate residual505.4125 versus385.5569ms. Exact events show18 fresh264MiB allocations/frees across18 warm+measured candidate requests. Thus extra time is outside modelprefill forward; fresh driver allocation is a credible cause, alongside policy placement/copies/controller setup. No per-allocation timer was recorded, and the twoepochs differ source guards/walltime, so the~170ms residual difference is not direct cuMemHostAlloc measurement. A CPU/GPU-context preflight explicitly timing existing allocate/complete release or instrumented allocation boundary is required before attributing cause quantitatively. Do not claim all residual is allocator work.

## Reuse existing code without changing controller semantics

No exactroot sequential-request owner/lease helper found. Existing reuse policies reuse visibleGPUdecode buffers; ExactSharedPool/ExactBulkPayloadPool own and free a root for each fresh manager. Reuse `exact_pinned_memory.allocate`, BulkPayloadPool operations, managerflush/synchronize and currentpolicy reset ordering; introduce only an explicit single-root owner and serial lease. No global size cache or thread-safe serving abstraction needed.

Allocate one exact264MiB root at declared server/sessionstartup on the intended CUDA ownercontext. The owner holds thattensor until explicit finalclose; each fresh manager obtains a same shape/dtype/context lease. Policy scores/CPUcanonical planes/logicalfree/shadowmaps remain newly initialized perrequest. Cold payload need not be zeroed if every newly live logicalslot is written before read and stale unused slots remain unreachable. Preserve existing stage/event ordering.

**Critical reset ordering:** `CpuBatchPolicy.reset` closes manager first, then superreset closes outgoing per-cache poolviews. Releasing an owner lease inside manager.close alone therefore releases it before all views clear. Release only after the complete policy.reset returns and managerflush/wait/cacheview cleanup finishes; reject concurrent acquire and different thread/context/geometry. If reset fails, keeplease quarantined/busy until cleanup succeeds—do not silently hand the root to another request. Freshmanager should retain BulkPayloadPool constructor/read/queue_bulk/flush semantics; owner only supplies allocation and tracks lease, not canonicalstate.

Acquire validates exact shape, BF16 dtype, resolveddevice and actual CUDAcontext identity; reject closedowner/busyroot/foreigncontext/unsupportedgeometry before exposing storage. Explicit owner.close rejects activelease, waits ownercontext and drops finalroot after aliases close. Existing allocator finalizer handles eventual free; verify stats show one liveallocation while provisioned/idle and zero after finalclose, with exactly one matching allocation/release eventpair. Do not fake zero afterrequest reset: root remains resident intentionally.

## Honest prospective experiment

Record server/startup allocation latency separately and also publish first coldrequest latency including provisioning. Steady requests may exclude that documented one-time startup only after explicitly naming this different deployment protocol; compare arms with same warmup/steadyrequest boundaries. Retain original coldTTFTfail. Record provisioned hostroot and publicallocationrange throughout idle and active intervals; include it in candidate resourceceil even between requests. Startup cannot erase capacity cost. Include model/staticGPU ownership, requestGPUpeaks, canonicalCPUplanes, descriptorreserve and cleanup controls as before; inspect hiddenhost ownership limitations unchanged.

Before timing: paired full-logit/payload currentnative parity, alternating sequential prompts including error/retry and changedshape rejection, no overlapping lease/context, stalepayload unreachable, manager/cache alias cleanup, exact live-root/accounting invariants, and fullguard tests. A mockowner CPU test can prove acquire/busy/reset/close ordering; native same-context integration still needs measuredGPU proof. Prototype can be artifact-only; do not alter ordinarypolicy or existing frozen source while current stressproofs run.
