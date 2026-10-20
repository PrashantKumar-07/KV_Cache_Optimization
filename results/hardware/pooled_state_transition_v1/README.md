# Pooled GPU state transition primitives

This is a distinct layout mechanism. Fixed shared GPU owners hold cumulative
scores, ages and cold timestamps; logical prefix views retain the original
owners. A fused prepare kernel increments old/promoted ages and initializes
new metadata. Compaction gathers BF16 KV and F32 metadata to separate bounded
scratch before scattering to consumed KV and the pooled metadata owner. Host
checks validate geometry, keep bounds, pitches and aliases before pointer
uploads. Source tensors and metadata owners record the active CUDA stream.

The CPU suite passes450 tests, including13 focused checks. Five GPU fixtures
check exact payload/metadata bits (including nonfinite bit patterns), owner
roots, padding and worker-stream behavior with explicit source dependency.
Preparation covers3 and32 layers; compaction covers1,3 and8 layers. Two malformed
native descriptor guards reject before dispatch. This does not test implicit
cross-stream readiness or a complete model controller.

For32 layers, hot capacity1055 and cold capacity1154, declared metadata storage
is417792bytes (0.3984375MiB). Eight-layer8-head991-row128-dimension compaction
uses32536512bytes (31.0292MiB) of bounded KV/metadata scratch. Host descriptor
uploads are384bytes for32-layer prepare and32192bytes for the measured eight-layer
compaction. These are declared primitive storage and helper upload counts,
not full-model peaks, full-process allocations or physical bus transactions.

The full controller integration is incomplete. The next path must replace
supported promotion/append metadata allocation, preserve score projection and
age ordering, and retain immediate cold-reclaim fallback. Then prove recorded
model prediction/state/visibility/counter/native KV equality and measure actual
resources and the preregistered whole-loop speed gate. A primitive check cannot
establish a joint latency/memory/quality win.

No production source, defaults or CLI changes. Failed wrapper implementations
stay stopped. This uses existing Torch/CUDA and compilers; no dependency is
installed. Execution sources, plans, logs and compiled binary hashes are retained.
All owned jobs are terminal. Physical STT, independent quality, comparable faithful
baselines and full P0/P1/P2 remain open; no completion day estimate is asserted.

See [verified stage](verified_stage.json), [raw GPU fixtures](gpu.json) and the
archived execution source manifest.
