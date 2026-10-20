# Shared-transition regression localization

Two fresh GPU0 runs use the same policy-only profiler and frozen 16K/50-step
prompt, teacher stream, geometry and model. Each exactly replays its frozen
predictions, final state, counters, KV payload and native transfer call totals.
Leases, attention hooks and owned pinned roots release. Execution sources and
plans are archived. This diagnostic does not retry or reverse the prior failed
quiet 3ABBA timing decision.

The control performs 50 cross-layer native gathers; the candidate performs 1600
per-layer gather-and-copy compactions. Candidate compaction is 6.76 instrumented
ms/decode versus 3.12 for ordinary compaction preparation plus 0.75 native gather
at visibility. Shared reservation adds 1.19 ms and hot extension is 1.88 ms;
these inclusive ranges overlap other calls. They cannot be summed as savings
or establish exclusive causation for the 8.98% uninstrumented slowdown.

The source trace shows cold reclaim uses compacted hot positions and cumulative
scores after demotion; it does not consume compacted hot KV. A distinct bounded
batched payload transition can therefore defer KV while preserving metadata
ordering. This is an implementable next mechanism, not an already proven win.

Ordinary policy preparation, eviction and visibility still require broader
controller consolidation to close the faithful-baseline gap. No production
source or defaults change. Independent quality, comparable baseline widths,
physical hardware writes/energy and full P0/P1/P2 remain open. No day estimate.

See [verified evidence](verified_diagnostic.json) and the two raw profile files.
