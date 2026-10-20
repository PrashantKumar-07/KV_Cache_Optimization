# Batched GPU native-query read indices

Research-only`experiments/gpu_query_read_indices.py`: validate and refresh all
layer source-block/live-logical maps once after ordinary visible preparation;
reuse the existing nativeH2D CUDA-index kernel. Selector AST changes only live
index lookup and paired full-live read. Q/ranking/top-k/row order stay unchanged.
Stale maps or unconsumed ordinary prefetches raise before native dispatch.

[Correctness](verified_correctness.json):10CPU map/range/geometry tests,
283suite pass. NativeBF16/FP16/F32 K/V exact, serial-consumer streams with
explicit caller handoff, two-pool registration switching, empty/stale/prefetch
cases and root/metadata cleanup pass. GPU4K12 pairs exactly replay frozen native
query12predictions/13logit-cache-counterframes/384visibleKV choices and all
nativeKVbytes. No canonicalCPU cold reads during selection.

[Corrected resource proof](verified_resource_v2.json):16K50 active envelope
547.488MiB vs544.321old query fits the576MiB pilot, not512MiB. Roundedownedhost
256MiB/reserved5089.492MiB/prefill18.62783GB unchanged. Full-query predictions,
state/counters/nativeKVbytes replay; ownroots/hooks/indexmetadata release.
NativeKVH2D5.825961984GB unchanged; candidate metadataH2D75.33312MB and live
GPUindexstorage1.409MiB are charged separately. Old generic index-metadata
copies are unmeasured, so there is no totalH2D saving claim. Original resource
report copied a stale development-readiness label; original verifier/report
remain immutable and the corrected proof permits component cost only.

[Component cost](verified_cost.json):3ABBA16K50/warmup1, originalquery vsGPU
read-index query. Median139.451→105.133ms,24.61%/34.317ms saved. p95 andTTFT
guards pass. Exact frozenquery replay,300measured forwards/arm,25quiet before/
afterGPU0 computePID observations and ownhook removal verified.

[OrdinaryLRU guard](verified_lru.json):a separate fresh3ABBA comparison gives
median58.215→105.362ms(1.810×), p9564.477→108.280ms(1.679×), both fail2%guard.
TTFT+0.378%passes. Eacharm replays its corresponding frozenstate/counters/
predictions/KVbytes;300measured forwards/arm and25quiet process observations.

**Stop standalone full-query GPU-index candidate expansion/adoption.** Keep
correctness and component speed evidence as research knowledge; no ordinary
CLI/default change or oldfullscan quality retry. Next inspect fused ranking
and lower-traffic retrieval with GPU-generated selected-row indices. The
failed fixed8-page centroid heuristic stays stopped. Full512MiB/actual
faithful-baseline matching, independentquality, physicalSTT and fullP0/P1/P2
remain open. GPU jobs42958/30311/37219/76898/59757 finish exit0.

[Diagnostic figure](read_index_tradeoff.pdf), [comparison data](read_index_tradeoff.csv).
Preregistrations and source archives precede execution; raw/proofs cannot be
overwritten. GPU process observations are before/after rows, not continuous
utilization monitoring; stream tests do not establish concurrent serving.
