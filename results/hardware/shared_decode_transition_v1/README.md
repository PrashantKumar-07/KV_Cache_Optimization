# Shared decode ownership: memory passed, timing failed

Unified ownership reuses the consumed visible buffer for the one-row model
append, policy promotions/appends and immediate hot compaction. All layers are
validated before mutation; unsupported cases use the existing fallback. Custom
instance update overrides are excluded. Original ranking, retention, native
kernels, production defaults and CLI behavior remain unchanged.

The final CPU checkpoint passes 385 tests, including 11 focused unified checks.
The recorded GPU 4K/12-step pair matches 12 predictions, 13 cache/visibility
frames, counters and native KV traffic/calls. Fresh 16K/50-step replay matches
the frozen ordinary predictions, final state, counters and native transfers.
There are 1600 shared bindings/compactions and fast appends, with zero fallback.

Scoped active memory is **415.163 MiB versus 541.077 MiB**, a **23.271% reduction**,
passing the preregistered 512 MiB OR 10% reduction gate. The scope includes the
owned 256 MiB pinned allocator block and excludes constant model storage,
unowned host allocations, non-Torch GPU allocations and driver/process memory.
The whole reserved envelope remains 5089.492 MiB; prefill peak is unchanged at
18.627831 GB. This is not a full-process 512 MiB claim.

The independently preregistered quiet 3ABBA comparison fails the whole-loop
speed gate: median **58.844 to 64.126 ms**, **8.976% slower / 5.282 ms added**;
p95 **64.553 to 69.319 ms**, **7.383% worse**. TTFT +0.783% passes. All 300
measured decode forwards per arm match frozen predictions, final state,
counters and native KV traffic. All 25 before/after GPU0 process checks pass.
Append counts include warmup (3200); policy counts describe the measured trial
(1600). Temporary leases, instance methods, hooks and pinned roots release.

**Stop expansion/adoption of this implementation.** No favorable timing retry,
wider quality run or faithful-baseline expansion follows this failed speed gate.
The next useful diagnostic is the added cost of immediate compaction and buffer
transitions. It must justify a distinct mechanism before another model screen.

Plans precede their runs, including the immediate-compaction amendment before
GPU measurement. Archives preserve execution sources; raw data, logs and
independent verifiers remain separate. The standalone model append failed its
memory gate and remains recorded in `../reserved_decode_append_v1/`.

Limits: one model, seed and fixed teacher; full-model logits were not separately
captured. GPU process checks are intermittent. Larger appends, concurrency,
independent quality, equal-width faithful baselines, hardware writes/energy and
physical STT remain unproven. Full P0/P1/P2 is incomplete. The unsupported
10–20-day forecast remains withdrawn.

See [correctness](verified_correctness.json), [resources](verified_resource.json),
[timing decision](verified_cost.json) and [figure](ownership_tradeoff.pdf).
