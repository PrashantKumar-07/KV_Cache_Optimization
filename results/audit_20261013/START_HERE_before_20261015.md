# Start here

Current work: [October audit and remediation](AUDIT_20261013.md). The audit is active; paper preparation is paused until correctness, reproducibility and improvement evaluation finish.

The user confirms STT-RAM is a **Python simulation**. Physical STT access is not required for this study. Keep assumed device costs separate from measured GPU/CPU time and logical events separate from physical writes/endurance.

The old GPU availability blocker cleared on 13 October. Before fixes, all 518 existing tests passed but eight independently reproduced core semantic checks failed. The fixes now pass those checks and 577 tests, including absolute-position/payload checks for bounded prefill. Historical prompt protocols and context-only TriviaQA document identities are invalid as current headline evidence.

Frozen native cold-boundary timing passes: 49.1732→44.2350 ms/token, 10.04% and 4.9382 ms saved versus its own bulk controller. P95 improves; TTFT rises 1.71% within the original guard. This is not a faithful-baseline win or a current-source adoption decision. [Frozen proof](results/hardware/cold_boundary_controller_v1/verified_cost.json). Corrected-source wider-context/generation/boundary paired validation passes (300 teacher steps and690 generated decode steps); current quality/improvement gates remain open. Artifacts in [new artifacts](results/hardware/cold_boundary_audit_20261013/).

Run CPU/tiny-model checks from this directory with `../venv_kvcache/bin/python -m pytest -q`. Use new output paths after source/protocol changes. Frozen raw evidence remains unchanged. The [previous start guide](results/audit_20261013/START_HERE_before.md) is historical.

Resume after interruption: read `AGENTS.md` and `TASK_STATE.json`, check listed processes/output logs, then follow the ordered pending gates. Do not duplicate GPU jobs or resume a checkpoint after source/runtime mismatch.

Actual-query prefill passes a consumed-development quality screen: +4.1646 points, document-cluster 95% interval [0.4847, 8.2478]. This warrants further evaluation; no independent-data, fair-resource or SOTA claim. [Verified raw quality](results/audit_20261013/actual_query_development/verified_quality.json). Grouped-head architecture pilot failed its registered gate after independent rescoring (−2.8091 points); stop expansion.

Four superseded root documents have a checksummed, round-trip verified [archive manifest](results/audit_20261013/documentation_archive/manifest.json). Originals remain until the user’s requested full review condition is met. Use these current entrypoints for active work.

The prior grouped-only zero-head guard passed557 tests. Further ordinary/batched reproduction found the same invalid-input issue in shared paths; validation now rejects malformed feedback before any score/LRU mutation across layers.73 focused and572 full checks pass. Frozen experiment source proofs remain separate from the live worktree.

Final prior-preserving adapted baseline reproduction has finished:90 predictions verified; Tiered−Snap+3.2332 with conditional row95%[-0.6913,8.8743], equalhot only. [Verified raw results](results/audit_20261013/final_prior_development/verified_quality.json).

Actual-query resource3ABBA screen completed and all registered guards pass; [verified resource proof](results/audit_20261013/actual_query_resources/verified_resources.json). No owned GPU jobs remain at this checkpoint. Query/timing runners now record model artifact identity. Four archived docs have full unique-prose review evidence, but project-wide review is incomplete, so their originals remain.

Current-model faithful SnapKV development comparison and full-logit/native-controller parity are complete. The faithful quality screen establishes no advantage (−0.2759 points, document-cluster95%[−1.8870,1.2861]); its matching452MiB boundary is initial held KV tensor storage only. Native parity verifies51 complete logit arrays and32-layer KV traces per arm on one16K teacher-forced input. Current controller timing has finished; independent proof is pending and preliminary scoped memory fails the registered512MiB guard. See TASK_STATE.json for exact artifacts and next work. No owned GPU jobs are running.

The exact-size pinned allocation follow-up has completed its raw strict full-model parity gate; independent verification is pending. Its resource screen retains the rejected predecessor’s original512MiB ceiling and performance limits. Source is frozen until measurement and verification finish. Check TASK_STATE.json for the actual running stage before issuing jobs.

Latest checkpoint: exact controller resource proof rejects TTFT (ratio1.12583>1.10) despite493.65MiB scopedmemory passing512MiB. The exact allocator isolatedprobe measures162.68ms freshallocation versus0.064ms warmedTorch allocation.577 fulltests pass after three hookconstructor rollback fixes; future explicitserial rootreuse is notimplemented/adopted.8Kcold256 stressoutputs are byteidentical but nativecoldcoverage41/50fails; corrected raw checking shows total cold occupancy reaches256 and oldcoldpositions are replaced (earlier reviewer conflatedlive/total).31.5K diagnostic failedpromptlength precondition beforeforwards. Preservebothfailures. See TASK_STATE.json for active fallbacklocalization stage and no paper/MDcleanup completion claim.
