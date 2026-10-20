# Frozen controller transition corpus

One ordinary-backend GPU run captures1600 layer transitions from the frozen
16K/50 teacher stream. Every prediction, final state, logical counter, native KV
payload and transfer-call count matches the existing ordinary reference.

Each record stores post-promotion/pre-eviction input, prepared victim decisions,
new token positions, and post-eviction output before visibility compaction.
Positions, hot grace, cold shadow/resident/streak flags, GPU F32 metadata bits,
physical logical/free slot ordering, pending compaction indices and outcomes
are retained. Typed arrays use base64 raw bytes; gzip is the transport.

Independent replay matches all1600 records:2560 paid rows,23872 restored
backups and1600 queued compactions. CPU flags, F32 metadata, slots/free order,
positions and outcomes are exact. The existing native backup classifier also
matches every record. All hooks/method wrappers restore and pinned memory
releases. Snapshot serialization and GPU synchronization invalidate capture
latency as a performance measurement.

Cold occupancy reaches897 of1090 rows; no overflow occurs. Saturated reclaim,
ragged/empty/tie/nonfinite/invalid-input fixtures, promotion/selection checks,
and integrated long-output preservation remain required. The corpus freezes
victim decisions; it does not independently prove their selection. No KV
contents, new controller, speed, quality, physical STT or completion claim.

[Verified corpus](verified_corpus.json), [compressed records](transitions.jsonl.gz).
Source archive and launch plan preserved. This allows CPU transition validation
without another model launch for each implementation change.
