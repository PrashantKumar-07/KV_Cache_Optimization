# Pooled controller: exact replay and memory pass; speed fails

Research integration of the verified fixed GPU metadata arena. Hot cumulative
scores/ages and cold timestamps retain shared owner roots. One cross-layer
prepare initializes tails and ages; bounded eight-layer compaction gathers
KV/F32 into scratch and scatters safely into consumed KV and pooled metadata.
Cold overflow retains immediate compaction before reclaim; unsupported inputs
use the existing path. No process-global production patch or new dependency.

- Full suite:459 passed, including9 focused controller checks.
- 4K/12:12 predictions,13 cache/visibility frames, counters/native KV exact;
  pooled prepare12/compact48; append384/fallback0.
- Fresh16K/50:all frozen predictions/state/counters/native KV exact;
  pooled prepare50/compact200; append1600/fallback0.
- Scoped active:ordinary541.077 MiB,candidate432.096 MiB; candidate432.268 MiB
  with conservative CPU array roots. Prefill18.627831 GB unchanged.
- Quiet3ABBA,300 forwards/arm:median58.061 to61.542 ms (5.996% slower);
  p95+0.667%,TTFT+0.873%. Both speed primaries fail. Stop expansion/adoption.

Arena417792 bytes; scratch32536512 bytes. Pooled descriptors6457600 logical
H2D bytes/850 uploads per50-step trial; timestamp ranking descriptors5555328
bytes/153 uploads. These are separate from KV traffic, not physical PCIe
transactions; GPU peaks and complete-loop timing charge this work.

[Correctness](verified_correctness.json), [resources](verified_resource.json),
[timing](verified_cost.json), [figure](ownership_tradeoff.png).
Plans precede launches; measured source archives and native binary hashes are
retained. Attempt1 exposed a generated-helper/local-variable collision, fixed
before successful model measurement. Attempt2 stopped at an existing partial
report guard; no model/timing dispatched. Original failure logs/reports remain.
All owned jobs are terminal; hook/lease restoration and pinned release pass.

One Mistral model/seed/fixed teacher. Fullmodel logits are not separately dumped.
Torch active peaks exclude constant model storage and include owned pinned
block; native containers/Python heap/allocator/non-Torch GPU/driver allocations
remain unmeasured. No independent quality, faithful-baseline win, physical STT,
full-process memory or full-project completion claim. Production defaults stay
unchanged; failed prior candidates remain stopped.
