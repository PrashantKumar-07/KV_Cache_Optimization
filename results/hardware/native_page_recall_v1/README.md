# Fixed eight-page recall screen

This read-only GPU audit follows the native full-query selector's latency
failure. It evaluates one fixed candidate rule before implementing an index:
stable groups of16 cold rows, eight pages ranked by maximum query-head centroid
score, and enough pages to fill the32-row quota.

The fresh16K/50-step native-query run exactly replays all50 teacher predictions,
final cache state and logical counters. Ownhooks and the canonical-read guard
restore.1600 layer-step observations yield:

| Diagnostic | Result | Preregistered screen |
| --- | ---: | ---: |
| Mean top32 row overlap |28.3945%|at least90%|
| Mean best selected attention mass |55.3975%|at least95%|
| Mean candidate fraction |13.4900%|at most25%|
| Underfilled selections |0|0|

**Stop implementation and wider quality/timing tests of this fixed8-page
heuristic.** This does not reject every possible pruning method.

Mass uses the best top32 under the full-key attention distribution inside the
candidate set: an optimistic ceiling. Candidate-only softmax can change
head-normalizers and ranking. These figures do not establish generation
quality, independent noninferiority or superiority.

The extra full-key audit reads5.5148544GB through native paired transfers;
total observed H2D is11.340816384GB including the unchanged selector. Candidate
payload0.74395648GB is projected from candidate counts; summary maintenance,
other transfers and ranking are omitted. It is **not measured transfer savings**.
Summaries are rebuilt offline; this is not a maintained native index.

Existing`StablePageIndex` reads canonical`cache.stt_k` in its constructor,
reconciliation and candidate-device selection. That property synchronizes the
native manager and materializes a canonical CPU tensor. Its existing GPU-cold
implementation cannot simply be enabled for the native CPU pool. The ordinary
CPU CLI's LRU guard remains unchanged. Historical indexed GPU-cold latency
also remains separate evidence, not a current native timing result.

[Preregistration](plan.json), [verified decision](verified_report.json),
[diagnostic figure](recall_screen.pdf), [per-layer data](recall_screen.csv).
Original/control and diagnostic sources are archived before launch. Job43414
finished with exit0; no ordinary code changed and existing266tests remain.

Next investigate the overhead of the current-query hook using already-visible
rows before proposing a new lower-traffic index mechanism. Full512MiB, actual
faithful-baseline matching, independent quality and physicalSTT remain open.
