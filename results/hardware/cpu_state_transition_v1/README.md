# Fixed CPU state phase: exact replay and GPU handoff

Shared NumPy owners hold hot/cold positions, grace, cold shadow/resident flags,
streaks and physical logical/free slot ordering. Native C++ promotion, append,
demotion and reclaim operations preserve these roots. Demotion computes backup
classification, GPU write/keep commands and CPU metadata changes together.
Reclaim preserves freed-slot FIFO ordering. Promotion commands own their rows,
counts, physical read slots and hot start. Commands are read-only and tied to
owner identity/version; consume them before a later phase reuses freed slots.

All489 tests pass, including30 focused checks for batch rejection before any
mutation, bounds/capacity, malformed owners/aliases/alignment, raw F32 bits,
ragged/empty input, live floor, saturation/reclaim, duplicate backup identities
and stale/foreign commands. First compilation failed at a mixed signed/unsigned
C++ auto declaration; the source/log is retained. Subsequent stage corrections
and successful source archives are retained; no timing measurements occurred.

The final native phase exactly replays all1600 frozen ordinary transitions in
50 batches:2560 paid rows and23872 restored backups. CPU metadata, GPU metadata
commands evaluated on recorded F32 arrays, positions and slot/free ordering are
exact. Owners remain stable. The interrupted final replay has no available
process exit code: complete matching output/log, a missing handle and absent
process establish termination and artifact success; it was not restarted.

Five GPU handoff fixtures across20 layers additionally consume the commands
using fixed GPU metadata and KV owners and compare original TieredKVCache.
They cover shadow-first overflow, stable LRU, live-floor cover edge, exclusive
promotion and eight-layer compaction. BF16 payload/F32 metadata, roots and
padding are exact. GPU slots emulate physical addressing; pinned host/native
transfer integration is not established. Victim decisions are supplied and
cold keep decisions come from the original reference, so retention selection
is not independently tested here.

Declared32-layer CPU owners:1116896 bytes (1.065 MiB). Native temporary vectors,
output descriptors, Python allocator and full-model memory remain unmeasured.
The recorded corpus has no overflow; synthetic fixtures cover that phase.
No whole-loop speed, quality, faithful-baseline or physical STT claim.

[Verified phase](verified_stage.json), [recorded replay](replay.json),
[GPU handoff](gpu.json). Plans, sources, binaries, raw jobs/logs remain archived.
**Controller integration is still incomplete.** Next attach these canonical
owners and consume the commands through the complete decode/pinned KV path,
then prove model replay/resources/quiet3ABBA before any adoption. Production
defaults and failed prior-candidate decisions remain unchanged. Full P0/P1/P2
remains open; the unsupported day estimate stays withdrawn.
