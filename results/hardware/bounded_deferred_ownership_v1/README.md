# Bounded deferred ownership: memory passed, timing failed

A distinct research transition defers hot compaction until the visibility
boundary when no cold overflow exists. Cold overflow and unsupported layouts
retain immediate compaction. Groups of at most eight layers gather into separate
scratch, then scatter back using two passes of the existing verified native
reuse kernel. Physical head pitch and all keep bounds are validated before
mutation; scratch protects against in-place gather races. Original ranking,
retention, native sources, production defaults and CLI behavior stay unchanged.

The final CPU suite passes 401 tests, including 16 focused checks. The initial
run found a missing logical-length guard and a fixture type error; both are
corrected, with original failed logs and a post-run/pre-correction source copy
retained. The GPU 4K/12 pair matches 12 predictions, 13 cache/visibility-state
frames, counters and native KV transfers. Fresh 16K/50 matches frozen predictions,
final state, counters and transfers; 1600 deferred compactions use 200 groups.
There are no immediate compactions or append fallbacks on this stream.
Scratch is bounded at 32,473,088 bytes (30.970 MiB).

Scoped active memory is **432.116 versus 541.077 MiB**, **20.138% lower**, passing
the 512 MiB OR 10% reduction gate. It includes the owned 256 MiB pinned allocator
block and temporary scratch. Constant model storage is excluded. Whole reserved
envelope remains 5089.492 MiB and prefill peak remains 18.627831 GB. This is not
full-process memory; unowned host, non-Torch GPU and driver allocations are excluded.

Quiet 3ABBA median decode is **58.331 to 63.086 ms (8.152% slower, 4.755 ms added)**;
p95 **64.770 to 70.877 ms (9.430% worse)**. TTFT +0.360% passes. All 300 measured
forwards per arm replay frozen behavior and KV traffic. All 25 before/after
GPU0 process observations pass; owned leases, methods, hooks and roots release.
**Stop this candidate's expansion/adoption.** No favorable timing retry follows.

The copied scalar `candidate_extra_metadata_transfers=0` in the original timing
proof is incorrect. A reporting-only correction retains the original proof and
raw benchmark. Fixed native helper geometry derives **12,864,000 descriptor H2D
bytes and 1200 upload calls per candidate measured trial**, separately from the
native KV manager payload/calls. Total extra descriptor traffic versus ordinary
was not independently instrumented; derived bytes are not physical PCIe transaction
measurements. Complete-loop timing and actual allocation peaks charge this work.
Use [the correction](verified_descriptor_reporting_v2.json) alongside
[the original timing decision](verified_cost.json).

Broader controller consolidation remains the next target. Bounded payload grouping
alone does not establish a joint memory/latency/quality win. Larger appends,
recycled requests, overflow GPU cases, concurrent serving, independent quality,
equal-width faithful baselines and physical STT remain outside this candidate's
proof. Full P0/P1/P2 stays incomplete and no day estimate is asserted.

See [correctness](verified_correctness.json), [resources](verified_resource.json),
[figure](ownership_tradeoff.pdf) and the separate archived execution sources.
