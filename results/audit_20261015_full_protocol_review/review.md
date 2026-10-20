# Full six-task rerun: independent protocol review

The current-source full rerun is justified as a descriptive reproduction and expansion, provided every prediction is newly generated under one frozen protocol. It is not evidence of six-task independent generalization or SOTA superiority. This review launches no GPU work and changes no product source.

## Exact scope

`original_population.json` independently maps every historical sample ID to the original source file. The original135 are indices0–24 for MultifieldQA, Qasper, NarrativeQA, HotpotQA and TriviaQA, and0–9 for GovReport. Population counts in that task order are150,224,200,300,300,300: total1474. All six file hashes match the historical configuration. The resolver prioritizes `_e` files (`experiments/longbench_eval.py:1281`), except NarrativeQA where only the standard file is used. Do not union standard/E variants or silently substitute different files.

## Preserve final-data restrictions

Freeze all candidate arms, source/model/runtime/dataset hashes, analysis, stopping rules and complete sample IDs before new outcomes are inspected. Refresh the conservative target-document inventory, which excludes prior recorded sample IDs and manifest selections across standard/E variants (`experiments/audit_unused_documents.py:42`). Stratify the population into historical135, other consumed/document-overlapping rows, and previously unconsumed target-document groups. The first two strata are development/descriptive evidence. Evaluate reserved/unconsumed groups only after an explicit final candidate lock; do not alter the candidate after viewing either stratum's new outcomes. If the candidate remains open to improvement, run the consumed partition first and keep the rest unexposed.

NarrativeQA has zero unconsumed target-document groups in the existing inventory, so a six-task independent holdout is unavailable in these files. A complete1474-row descriptive table and a separate five-task document-disjoint final table are compatible; the combined score is never called an independent holdout. Report the original135 subset separately for reproducibility, without selecting it as the headline because it performs better. Consumed failed holdouts remain consumed. Hash/normalized-document disjointness does not establish semantic or model-training disjointness.

## Common controls

Use the same frozen Mistral weights/tokenizer, BF16, seed0, batch1, prompt templates and task chat exclusions, middle truncation31500, absolute position IDs, per-task generation caps, greedy decoding/EOS, first-line trimming where applicable and answer-max metrics. Record exact prompt/token hashes, token predictions, prompt/generated lengths and unrounded sample scores. Use full uncompressed prefill with SDPA and last-logit slicing for every arm; use eager decode for every arm. Required feedback weights may differ by algorithm but must be disclosed. One fresh run of unchanged controls may populate both hot-budget panels; archived control predictions may not substitute for the new run. Generate paired rows for every arm; errors/OOMs must be explicit, never silently dropped from complete-case reports.

Quality from free generation is separate from causal speed claims: generated lengths differ. Cost totals must be labeled analytical or measured, per-request/per-token, and decode/prefill separately. Concurrent GPU quality timing is descriptive. Preserve the already failed faithful-Snap quiet timing gates; this larger quality run cannot overturn them.

## Baseline fidelity and fairness matrix

| Arm | Implemented behavior | What can be compared | Limitation |
|---|---|---|---|
| Full | Full retained KV, no eviction (`longbench_eval.py:154`) | Same-model reference quality | Unconstrained memory; not guaranteed mathematical accuracy upper bound |
| Streaming policy | Four sinks plus recent rows; true absolute RoPE positions (`:166`, `:1120`) | Bounded1024-row sink/window control | StreamingLLM-inspired adaptation, no upstream rolling-position implementation proof |
| Adapted H2O | Key-as-query unmasked prompt initialization, head-averaged cumulative decode retention (`:196`, `:229`) | Historical bounded1024-row functional control | Not faithful upstream H2O; prompt scoring differs |
| Adapted SnapKV | Key proxy, head-averaged pooled prompt selection, sink/recent pinning and H2O bounded decode (`:346`) | Historical bounded1024-row functional control | Not upstream SnapKV; cannot support a faithful-Snap SOTA claim |
| Tiered hot1024 | Actual-query prefill; hot1024 plus cold2048 capacity, quota32 | Equal nominal hot-row budget with adapted controls | Up to1056 cached rows attended; extra cold capacity, staging and metadata; unequal total memory |
| Tiered hot992 | Same mechanism,992hot plus quota32cold | At most1024 cached rows before append, matching adapted controls' cap | Cold availability may reduce width; extra cold memory remains; match actual observed width including current append |
| Faithful Snap adapter1024 | Vendored selector with actual queries, per-head expanded32-head KV; prompt1024 then decode growth (`experiments/snapkv_adapter.py:22`, `:82`) | Fidelity baseline under same model/prompt/scoring | Fourfold KV-head storage vs8-head bounded controls and growing attended width; no exact resource match |

Do not call the hot1024 panel equal VRAM ownership or the hot992 panel equal total memory. Label the shared logical budgets and plot actual geometry/resources separately. Faithful Snap is an additional explicit comparator, not a relabeling of adapted Snap. Existing `evaluate_snapkv_adapter.py:15` uses the model's backend directly, so the new runner must enforce the stated prefill/decode backend contract instead of silently inheriting its defaultSDPA decode.

## Analysis to freeze

Task-balanced six-task mean is the primary descriptive aggregate; provide all task scores and sample counts. Recompute from unrounded predictions and shared metric functions, not rounded aggregate fields. Predeclare document-group paired bootstrap (fixed seed/replicate count), with target-document rather than few-shot context identity for TriviaQA. State intervals are conditional on this fixed source/candidate and do not erase prior tuning. Retain all arm comparisons and negative outcomes. Register any final superiority/noninferiority criterion before unblinding, including multiplicity treatment; do not select the most favorable baseline, capacity, task subset or interval after seeing results.
