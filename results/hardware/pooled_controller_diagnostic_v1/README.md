# Remaining controller cost: paired diagnostic

Both fresh GPU policy profiles replay all50 frozen predictions, state, counters,
KV payload and native transfer calls. Attention hooks, candidate append leases
and owned pinned memory release. This is diagnostic instrumentation, not a
retry of the failed pooled-controller speed decision.

The candidate's first pooled extension load takes0.836 instrumented seconds.
That cold-start load is already excluded by warmup in the acceptance benchmark;
removing it cannot explain or reverse the measured6.00% slowdown.

Native backup classification enters1600 times and consumes3.009 inclusive
instrumented ms/step. Isolated backup batching does not justify a4ms speed
candidate: the computation still remains. Broader eviction, state updates and
visibility work remain. Inclusive profile ranges overlap and cannot be summed
as savings. Profiles include CUDA waits and instrumented Python overhead.

Both arms have zero saturated cold layers on all50 steps; neither run verifies
cold-overflow behavior. The next design targets the whole CPU state-transition
phase with shared canonical owners, not one helper call. Frozen transition
inputs/outputs are captured separately for native stage validation.

[Verified diagnostic](verified_diagnostic.json). Source archives/plans/raw
profiles retained. Production defaults unchanged; prior failed candidates stay
stopped. Full P0/P1/P2 remains incomplete. No day estimate or quality/physical
hardware claim.
