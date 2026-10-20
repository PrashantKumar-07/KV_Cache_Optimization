# Query-phase diagnostic summary

Research diagnostic only. **287 tests pass**; four new CPU parity cases preserve
logits, cache state, counters and visible selections. Annotations unwrap to the
exact GPU-index selector AST. The isolated 16K/50-step run completed frozen-query
prediction/state/counter/KV-byte/call assertions and all five quiet GPU checks.
Own hooks and GPU index metadata were released.

The original profiler jobs remain failed: one concurrent pytest CUDA process
was rejected before warmup; two isolated jobs failed a name-only event-count
assertion. The retained v4 summary has **eight CPU phases, 1,600 observations
each, plus six duplicate GPU annotation spans**. The new read-only scope verifier
checks those rows without changing measurements or running another model.
Original failure reports, source archives and corrections remain intact.

| Phase | Instrumented CPU elapsed / decode (ms) | CPU self / decode (ms) |
|---|---:|---:|
| live index lookup | 3.225 | 2.823 |
| query projection rotary | 7.248 | 2.613 |
| fast key materialization | 7.629 | 5.292 |
| native cold pair read | 8.780 | 5.914 |
| attention ranking | 8.135 | 2.333 |
| topk cpu ids | 5.714 | 1.324 |
| gpu visible composition | 15.887 | 9.363 |
| cpu visible metadata | 0.711 | 0.711 |

Visible-K/V composition is the largest measured CPU range: **15.887 ms/decode**,
27.71% of the eight CPU phase ranges. This includes profiler overhead, launch
cost and waiting; it is **not an expected speedup** or an uninstrumented latency.
CPU and GPU times overlap and must not be added. GPU annotation spans include
launch gaps; Torch-correlated kernel sums incompletely attribute the native
transfer extension. Chrome trace export did not execute before the assertion.
No complete GPU-kernel profile or speed/adoption claim is supported here.

Next: preregister reuse of existing visible buffers with cold-row-only updates.
Inspect hot/cold aliasing and preserve causal width, chosen order, predictions,
logical metadata, transfers and cleanup before any component timing. An ordinary
LRU guard is still required; the previous standalone fullscan candidate remains
stopped at 1.810× median / 1.679× p95. Full cold KV traffic is unchanged at
5.826 GB, plus 75.33 MB of separately charged candidate index metadata.
Full 512 MiB, matched actual baselines, independent quality, physical STT and
remaining P0/P1/P2 work remain open.

Evidence: [verified summary](verified_summary.json), [phase CSV](phase_costs.csv),
[raw v4 failure report](profile_v4.json),
[corrected annotation scope](annotation_scope_correction_v2.json).
