# Start here

Read [PROJECT_STATUS.md](PROJECT_STATUS.md) for the current state and full remaining requirements. **Project incomplete; execution blocked; no supported day estimate.**

Required quiet GPU testing is unavailable: another user’s GPU0 process remains live after three goal turns, and GPU1 is occupied. Physical STT access remains unspecified. [Blocking audit](results/status_audit_20261003_blocked/blocking_audit.json).

1. Integrated bulk controller: **518 tests pass**, exact tiny GPU and4K/12,16K/50 model replay; latest native cold-boundary scoped memory **424.15MiB**, strict GPU decode unchanged versus bulk control.
2. Earlier early-shared faithful-baseline speed **fails**: **54.59 versus25.25 ms/token**; median/p95 gates fail. Candidate expansion/adoption stopped.
3. Initial KV matches341 MiB; full resource trajectories remain unequal. Physical STT measurements, independent quality, generality and final delivery remain open.
4. Same-control macro passes: **59.92→53.62ms/token,10.52%,6.30ms saved**, p95 and TTFT pass. 8K/31.5K/recycling stress passes, including307 reclaims. Free generation passes700 steps/12 development documents; exclusive/floor/write-aware boundaries also pass. Separate-process reproduction also passes **8.91%,5.48ms saved**. Native cold-boundary integration now passes exact replay and resource checks. Next speed gate waits for quiet GPU0: unrelated process rejected the attempt before model load; zero measured rows. Baseline/resource and independent evidence gates stay open. [Stage proof](results/hardware/bulk_payload_phase_v1/verified_stage.json).

Current integrated timing: [proof](results/hardware/bulk_payload_controller_v1/verified_cost.json).

Earlier faithful-baseline evidence: [resource](results/hardware/early_shared_cpu_phase_v1/verified_resources.json), [timing](results/hardware/early_shared_cpu_phase_v1/verified_cost.json), [figures](results/hardware/early_shared_cpu_phase_v1/figures/manifest.json), [requirement ledger](results/status_audit_20261003_quality_readiness/completion_ledger.json).

[Original start guide and history](results/status_archive/20261003/START_HERE.md) preserves all previous links, commands and details. Current source/runtime and frozen raw proofs take precedence over historical prose. Follow [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md); failed experiments stay stopped and the consumed final holdout cannot become tuning data.
