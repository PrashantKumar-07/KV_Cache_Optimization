# Native controller prototype: memory passed, timing failed

The research CPU module uses existing Torch bundled pybind headers and system
g++. It adds no dependency or production default. It calculates promotion
score/index ordering, hysteresis/floor transitions, last-shadow backup lookup
and physical address expansion. Build manifests validate source, Python ABI
and binary hash. The policy composes these calculations with bounded deferred
ownership and groups GPU timestamp ranking by identical live length. Existing
verified native gather copies timestamp bits as two BF16 lanes; the original
unstable CUDA sort ranks the recovered F32 values. Promotion and address phases
are grouped; backup lookup still enters native code per layer. Full transition
pooling is not implemented.

The CPU core passes 29 focused checks; integrated checks total36. The full suite
passes437 tests. Grouped/separate GPU unstable sorting matches44 fixtures,
including ties and nonfinite values. The recorded4K/12 pair matches12 predictions,
13 cache/visibility-state frames, counters and native KV transfers. Fresh16K/50
matches frozen predictions, final state, counters and KV traffic. Native plans50
have no fallback, backup classifications1600, address calculations101 and
rank groups51. This is one model/seed/teacher, not broad preservation proof.

Scoped active allocation is432.116 MiB versus ordinary541.077 MiB (20.138% lower),
passing512 MiB. Adding conservative CPU streak/address array roots gives432.288
MiB; native container workspace and C++/Python allocator internals are unmeasured.
The owned256 MiB pinned allocator block is included; constant model storage,
unowned process/driver and non-Torch allocations are excluded. Whole reserved
envelope5089.492 MiB and prefill peak18.627831 GB remain unchanged.

Rank descriptors upload5,555,328 bytes across153 helper upload calls per measured
trial. Bounded ownership descriptors derive12,864,000 bytes/1200 upload calls.
These are separate from native KV manager payload/calls; timing and GPU peaks
charge them. They are not independently measured physical PCIe transactions,
and total extra metadata versus ordinary is not independently instrumented.

Quiet3ABBA median decode57.923 to62.805 ms (8.428% slower,4.882 ms added), p95
64.383 to66.792 ms (3.742% worse), TTFT+0.247% passes. All300 measured forwards
per arm replay frozen predictions, final state, counters and KV traffic;
all25 before/after GPU0 process observations pass. Owned methods/hooks/leases
and pinned roots release. **Stop expansion/adoption** without favorable timing
retry. No wider quality/faithful-baseline run follows this failed speed primary.

The native calculations remain reusable research code. Wrapping the existing
layer state has not closed the full controller gate. A distinct complete
hot/cold state layout must justify its complete cost and preservation before
another model screen, including retention/quality and current faithful baseline
comparison. Physical hardware writes/energy, generality and full P0/P1/P2 remain
open. No day estimate or full-process memory claim is asserted.

See [correctness](verified_correctness.json), [resources](verified_resource.json),
[timing](verified_cost.json), [figure](ownership_tradeoff.pdf) and archived sources.
