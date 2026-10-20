# Exact-root controller parity prerequisite

Independent CPU verification passed; no model/GPU rerun or source/raw edit. Runnable command: `../venv_kvcache/bin/python results/audit_20261013/exact_controller/verify_parity.py`; exit0. Proof is `exact_controller/verified_parity.json`, log `verification.log`; verifier refuses overwrite.

All5914 frozen source files match archive/plan/live source; original runner, prompt, teacher stream and runtime match. Independently rehashed12 model/tokenizer artifacts including allthree weight shards. All51 full-vocabulary logits arrays[1,1,32000]FP32 are finite and byte-identical. All51 trace points×32 layers match full hot/cold/visible K/V hashes and position/shadow state. Fifty decode predictions independently recomputed from logitsargmax match raw records. Candidate nativecold50/CPUappend50/fallback0; successful hook counts restored.

Both arms report zero live exact allocations and requested bytes after reset. Candidate's single exact276,824,064B allocation/release event pair has identical token/size; control has no exact events. ExitStack now guards acquired hooks/leases/reset. Partial constructor failures are not exercised, and no general concurrent/multiple-context lifetime guarantee follows.

This is one reused16K teacher-forced input and50steps, not free-generation quality/generalization. Ages/cumulative scores/streak/native owner state are untraced. Diagnostic payload copies/synchronizations invalidate timing/traffic interpretation; serving timing and public observed allocation ranges require the separately registered resource experiment. The previous cached-pool741.65MiB rejection stays unchanged. No exact-root memory/speed gate pass or default adoption is inferred here.
