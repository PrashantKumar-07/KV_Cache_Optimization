# Reusing the native Mistral query projection

Research-only module:`experiments/reused_query_projection.py`. Scoped attention
hooks capture the actual model q-projection output and run cold selection
before nativeKV update. An AST transform changes exactly one expression in
the original paired-read selector; rotary/softmax/ranking/selection stay intact.
Pending context clears on normal/error paths; partial constructor removes its
already-installed hooks. NativeMistralAttention only; ordinary CLIs unchanged.

[Correctness proof](verified_correctness.json):7focusedCPU tests/273suite pass.
Paired fullGPU4K12 nativeCPU implementations exactly match12predictions,
13logit/cache/counter frames,384visibleKV/choice observations and native
H2D1379229696/D2H111562752bytes. Current control exactly replays the frozen
native-query run. Model query projection calls fall768→384. No canonicalCPU
reads inside selection; pending context and ownhooks clear.

[Component cost decision](verified_cost.json):3ABBA16K50/warmup1, original
visible-only duplicated-Q no-op hook versus projection-reused visible-only
hook. Both preserve ordinaryLRU predictions/finalstate/alllogicalcounters/
transfers and read no additional coldKV.300measured forwards/arm and25quiet
before/afterGPU0 compute-process observations; ownedhooks removed.

| Hook diagnostic | Duplicated Q | Reused native Q |
| --- | ---: | ---: |
| Median decode ms |82.1289|81.5589|
| p95 decode ms |87.8936|87.3507|
| Median TTFT ms |2192.0386|2194.3485|

Median saving0.5700ms/0.6940% fails the predeclared at-least5% primary; p95 and
TTFT guards pass. **Stop standalone projection-reuse expansion.** Correct reuse
can inform future research, but this adds no ordinary feature, full-scan
quality/timing retry, latency-win or adoption claim. The control in this table
is the duplicated-Q hook, not ordinaryLRU with no retrieval hook.

Next inspect fused ranking and GPU-generated native read indices. This result
does not measure actual resource scope or prove independent quality. Full512MiB,
matched faithful-baseline allocation/geometry/quality/latency, physicalSTT and
fullP0/P1/P2 remain open. GPU jobs32672/22096 finish exit0. All source/plan/raw
archives remain immutable; verification scripts cannot overwrite prior proofs.
