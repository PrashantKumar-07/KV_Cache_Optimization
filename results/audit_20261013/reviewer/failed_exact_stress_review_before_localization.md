# Independent failed-stress diagnostics

CPU-only verification completed, no experimental retry/source edits. Runnable `verify_failed_exact_stress.py` and log are adjacent; proof `verified_failed_exact_stress.json`. Both registered stress gates remain rejected; **no `verified_parity.json` pass was emitted**.

Independent source attestation `failed_stress_source_attestation.json` at07:08:14Z binds all5914 archive/plan/live sources plus both raw/runner/prompt/teacher/runtime identities before subsequent hook edits. All12model/tokenizer artifacts independently rehashed. Later live source changes do not relabel these frozen runs.

**8192/cold256:**51 finite full-vocabulary FP32 arrays match byte-for-byte;50 predictions independently recomputed from decode-logitargmax;51×32 fullhot/cold/visibleKVhashes/positions/shadows match. CPUappend50/fallback0, hookcountsclosed and exactrootzeroafterreset hold. But nativecoldvisibility steps41 instead of registered50 fails pathcoverage. Nine nonnative visibility steps are inferred from counters, not explicit saved rejection reasons.

All recorded post-state layers retain1024hot rows; minimumlivecold acrosslayers rises192→242 one row eachstep, always≥32. Thus no empty-live explanation, no changing32-row visibility quota, no inferred rawwidth rise>1, and no cold256 capacity overflow/reclamation are supported by these traces. Do not label the run demonstrated coldrecycling or assign fallback cause to exposuregrowth. Geometry/stride/owner/pending-state/version/rejection-branch data are not saved; safe fallback may be another supported guard, but rootcause is unlocalized. Exact observed outputs establish exercised output equivalence, not full native-path coverage or universal correctness.

**31500/cold2048:**frozen prompt is30907tokens, independently reproduced using recordedlocaltokenizer. `profile_one` rejects requested31500 before model-prefill forward; rawarms empty, no logitsfiles. This is a harness precondition failure, not quality/latency/modelcorrectness evidence. Source has ExitStack cleanup, but successfulrowhook/allocation counters were not saved, so runtimecleanup is not independently proved.

Both failures/rawlogs remain immutable. Do not loosen coverageguard, retry for favorable results, or infer serving timing from diagnostic hash synchronization. Any future failure-branch capture or sufficiently long prompt requires a separately named protocol/source epoch and preserves these outcomes.
