# Visible cold-suffix composition

Research implementation: [query_visible_suffix.py](../../../experiments/query_visible_suffix.py).
Only the composition block differs from the GPU-index full-query hook. Existing
hot-prefix values and destination storage remain; selected cold rows are written
with two index_select out operations. Q, rotary, float ranking, topk, CPU IDs,
relative indices, metadata and native reader are unchanged. No ordinary CLI flag.

- **306 tests pass**, including19 new bitwise ownership/stride/offset/guard and
  four real-Mistral parity cases.
- GPU4K12: exact frozen predictions/logits/state/counters/384visibleKV selections,
  384retained destinations, unchanged canonical hotKV and prefix, no canonical
  CPU reads, unchanged transfer metadata and ownhook/index cleanup.
- Fresh16K50: **545.17MiB** active versus547.49MiB concat;576pilot passes but
  full512fails. Roundedhost256MiB, reserved5089.49MiB and prefill18.62783GB remain.
  All frozenstate/predictions/counters/nativebytes/calls/indexmetadata replay;
  roots/hooks/index metadata release. Three omitted native source files were
  supplied in a supplement matching recorded AND frozen hashes; old archive/
  verifier retained, no resource rerun.

| Separate3ABBA comparison | Median control→candidate | p95 control→candidate | TTFT change |
|---|---:|---:|---:|
| Query concat vs suffix composition |105.32→97.05ms (7.85%faster)|108.21→99.88ms|+0.34%|
| OrdinaryLRU vs suffix query |57.87→97.56ms (1.686×slower)|64.51→100.59ms (1.559×)|+0.46%|

Component primary5%/4ms and p95/TTFT guards pass; ordinaryLRU median/p95 guards
fail. Each comparison covers300measured forwards/arm and25quietbefore/after
GPU0computePID observations; frozen state/prediction/counter/bytes/calls/metadata
replay and ownhook cleanup pass. This is one model/seed fixedteacher evidence,
not independent quality or continuous utilization proof.

**Stop standalone suffix fullquery expansion/adoption.** NativeKVH2D still
5.826GB and candidate index metadata75.33MB/50steps. No totalH2D saving, physical
STT, complete512MiB, actual faithfulbaseline match or current joint win is proved.
Next preregister a distinct keys-only ranking/selected-value reader with unchanged
query math. The old fullscan quality retry and fixed8-page heuristic stay stopped.

Evidence: [correctness](verified_correctness.json), [resource v2](verified_resource_v2.json),
[component](verified_cost.json), [LRU stop](verified_lru.json),
[standalone figure](suffix_tradeoff.pdf), [CSV](suffix_tradeoff.csv).
All four ownGPU jobs are terminal. Full P0/P1/P2 remains open.
