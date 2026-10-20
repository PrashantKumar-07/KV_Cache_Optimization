# Current-query hook cost diagnostic

ThreeABBA blocks compare ordinary nativeBulkLRU against a no-op query-ranking
hook using only the32 already-visible cold rows. It repeats the current
q-projection/rotary/head-fold/softmax/top-k/host-ID/visible-concatenation work,
selects all32 rows in their original order, and adds no cold reads.

Every measured row exactly replays ordinary50 teacher predictions, final
cache state, all logical counters and nativeH2D/D2H bytes. Ownhooks release;
25before/afterGPU0 compute PID observations are quiet. Job26977 exits0.

| Three-block aggregate | Ordinary LRU | No-op query hook | Ratio |
| --- | ---: | ---: | ---: |
| Median decode ms |58.5813|82.3664|1.4060|
| p95 decode ms |64.3552|87.3897|1.3579|
| Median TTFT ms |2177.8145|2192.5671|1.0068|

**The existing unfused per-layer hook pattern exceeds the2% decode regression
screen even without extra cold reads.** Reduced transfers alone cannot rescue
this particular pattern. Inspect projection reuse, fused ranking and direct
GPU read-index handling before another index mechanism.

This is a diagnostic, not a working retrieval implementation, actual-resource
audit or formal universal lower bound. It includes KV concatenation/sorting
bookkeeping; other fused or reused attention implementations can have different
costs. No independent quality, baseline joint win, complete512MiB, physicalSTT
or ordinary adoption claim. Existing266test validation remains unchanged.

[Preregistration](plan.json), [verified report](verified_report.json).
Existing profiler, warmup, native model, teacher input, quiet checks and ABBA
scopes are reused from the frozen query selector experiment. Only the research
hook and post-measurement state verification differ. Source archives precede
execution; failed experimental results cannot be overwritten.
