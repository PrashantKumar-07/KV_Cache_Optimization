# Standalone reserved model append: stopped

The one-row inference append lease and shared hot prefix preserve the recorded
4K/12-step behavior and the frozen 16K/50-step predictions, final state,
counters and native KV transfers. The CPU checkpoint passed 374 tests, including
16 focused checks. Scoped active memory is 530.995 MiB versus 541.077 MiB,
only 1.863% lower. It fails the preregistered 512 MiB OR 10% reduction gate.
No timing or quality expansion is authorized by that gate; this candidate stops.

Sources, plans, raw runs and verification are retained. The later unified
transition is a distinct model-and-policy ownership mechanism. Neither candidate
changes production defaults. Scope excludes custom instance update overrides,
training, concurrent serving, full-process memory and physical STT evidence.
The first two failed CPU fixture runs retain logs; their original execution
source snapshots were not captured. Successful execution sources are archived.

See [correctness](verified_correctness.json) and [resource decision](verified_resource.json).
