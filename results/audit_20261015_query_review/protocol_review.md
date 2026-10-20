# Actual-query prefill plus current-query retrieval: preflight review

Decision: no correctness blocker established for the proposed **ordinary GPU BF16, batch-one Mistral, full-prefill, full-live-cold query scan** development screen. Existing implementation is usable within this scope. Do not infer a native-controller, speed, independent-quality or equal-whole-resource result from the screen.

The concrete changed mechanism is decode cold exposure: same actual-query prefill and hot/cold retention policy; replace recency exposure with current-layer-query ranking over all live cold candidates. This is distinct from the rejected head-grouped retention experiment and does not revive that candidate.

## Executable preflight proof

`check_selector.py` ran entirely on CPU with an actual two-layer GQA Mistral, both FP32 and BF16,24 prompt tokens and8decode steps per precision. Both modes pass16layer observations against an independently expanded-query-head attention reference. Chosen cold positions and actual prepared K/V payloads match exactly; the hook observes the cache before current-token append and preserves shared mask length.15of16observations in each precision exercise stale-index position fallback. Real feedback dictionaries and cold timestamps map to the newly chosen positions correctly. All observer/capture/selector hooks close and head-group settings return unchanged. Raw result and source hashes: `selector_checks.json`.

## Code review conclusions

- **GQA math:** `evaluate_query_selection.py:97` reshapes query heads into `[KV heads, groups, head_dim]`; contiguous repeat-KV grouping agrees with Mistral. The independent expanded-head calculation matches chosen ranks in the probe. The current model has no query normalization layer; this hook is not a generic adapter for models with Q normalization, attention-specific biases or alternative rotary processing.
- **RoPE:** existing `position_embeddings` come from explicit absolute decode positions; the same rotate-half expression is applied to the projected current query. Stored keys already contain their own rotary positions.
- **Preappend ordering:** selector is a self-attention forward pre-hook and replaces only old visible past K/V. The actual attention module then appends the current token. Keeping equal visible length preserves the precomputed causal mask. The32cold exposure quota is shared by count, not by identity, across layers.
- **Stale `_vis_stt_idx`:** QuerySelector changes `_vis_stt_pos` but does not update indices. `longbench_eval.py:1064` validates all old index→position pairs; if any differs, it resolves positions, updates correct LRU rows, and leaves `_stt_attn_idx` absent. `_promotion_plan` then uses position mapping. This is slower but correct in the examined ordinary path; do not bypass the fallback in a new coordinator.
- **Small-live case:** when all live rows already fit the exposure count, the hook correctly leaves prepared visibility unchanged. Absence of selector activity on such rows must be reported as no-op, not a retrieval improvement.
- **Prompt/decode separation:** before policy initialization there are no layer caches, so selector returns during prefill. PromptQueryCapture supplies actual prefill queries. Full prefill with recent128 avoids the zero-window slicing edge case; bounded-prefill/query history is explicitly unsupported.
- **Scope:** direct `cache.stt_k` indexing means this QuerySelector must not be installed on pinned-CPU backends or FP8 cold storage. Use ordinary GPU same-dtype BF16 caches only. Keep `candidate_pages=0,use_index=False` so no centroid/index approximation enters the one-mechanism screen.
- **Heuristic boundary:** selection scores normalize over hot+all candidate live *past* keys, excluding the current token. Mean head-normalized mass is a current-query retrieval heuristic, not the exact attention mass the eventual restricted model forward will assign. Ranking and retention are shared across KV heads. Do not call this exact full-attention selection.
- **Snap interaction:** SnapKVAdapter temporarily sets `num_key_value_groups=1` because its cache expands heads. It must be closed before any Tiered arm runs on the same model; assert original group counts and hook counts at every arm boundary. A live Snap adapter would invalidate Tiered query reshape/attention behavior.

## Protocol requirements before90predictions

1. Freeze current source/model/tokenizer/runtime and runner. Three arms use identical30already-consumeddevelopment records (five per six tasks), exact prompt/token hashes, task-specific generation limits, seed and greedy decoding. Use fresh outputs; do not mix historical model/source epochs into paired effects.
2. Control: actual-query prefill + ordinary GPU recency exposure. Candidate: same policy + existing full-live current-query selector. Faithful comparator: Snap904/window32/expanded-head growing decode with `reuse_projection=True`. Close every hook/cache wrapper per arm and restore head-group settings.
3. Register task-balanced candidate-minus-control minimum+2points **and** within-task target-document-cluster bootstrap95%lower bound>0; require candidate-minus-faithful-Snap paired point estimate>0 as a separate advancement screen. State the bootstrap seed/resamples and cluster identity rule in advance. Preserve failure/no-advancement outcomes; no favorable reruns or selective task dropping.
4. Record selector calls, selected layer count, live rows scanned, and small-live no-ops using observation wrappers that do not alter selection. This demonstrates the changed mechanism actually executes. Also record token counts, KV configuration, quota and resident trajectory where available. Do not interpret quality-only tracing runtime as benchmark timing.
5. This compares known consumed development records. Its bootstrap uncertainty is conditional on those selected records, not an independent generalization result. Passing only advances to a locked candidate, native integration parity and resource/full-loop cost gates before genuinely unused document-group quality evaluation. Extra all-live scoring and K/V copies are charged later, not omitted by claim.
6. The historical452MiB equality was initial held-KV-only and source-specific. Current-query hook creates replacement visible buffers and additional temporary scoring tensors; report GPU/CPU and full resource scope separately. Do not call this equal total memory or equal attention width versus growing Snap.

## Closed failed experiment

`results/audit_20261013/reviewer/grouped_pilot_verified.json` records the distinct head-grouped retention candidate,6consumed development pairs, task-balanced−2.809062points against registered+3; decision “reject; stop candidate expansion”. Keep it rejected; this review neither repeats it nor changes its mechanism/threshold.

No product source edits or GPU jobs were performed. This preflight is independent code/tiny-model review, not independent quality validation of the proposed30development pairs.
