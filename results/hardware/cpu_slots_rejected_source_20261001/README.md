# Rejected native host-slot real-cache trial

Exact executed sources are preserved here for inspection. The complete
src/experiments/tests source bundle is ../cpu_slots_executed_source_20261001.tar.gz.
Raw results are ../cpu_slots_abba_16k_50_20261001.json and the linked
synthetic/4K/8K/16K parity JSON files. Kernel binaries and source provenance
are frozen in results/baseline_fidelity/flexicache_transfer_check_v2_20261001.json.

Three ABBA blocks retained exact placement/state/output/counter/payload parity.
Pooled median 89.228 ms CPU-fast vs 88.918 ms slots (0.35% faster), below
5% acceptance; p95 110.828 vs 101.850 ms. The native pool reserves
402,784,256 bytes while 212,344,832 bytes are occupied after 50 steps.
The isolated storage gain did not establish a useful whole-loop improvement.

The candidate and its storage hooks were removed from the active project.
The standard CPU-fast experiment remains runnable. This archive needs the
recorded external native kernels, verified local model, prompt/teacher-stream
artifacts and project Python environment for replay; it is not an installed
production feature. No STT bandwidth, selector or default-policy change.
