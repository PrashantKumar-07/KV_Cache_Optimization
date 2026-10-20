# Implementation and evidence scope

The project studies a bounded hot KV cache backed by a recoverable cold tier.
The implementation and experimental evidence have several distinct scopes.
Read [the README](../README.md) for installation and usage and
[START_HERE.md](../START_HERE.md) for the current research checkpoint. This
document explains the mechanism; it does not establish a SOTA result.

## Executed model policy

`experiments/longbench_eval.py::TieredKVPolicy` integrates the per-layer cache in
`src/tiered_kv_cache.py` with model-generated keys, values, and attention.

1. After prefill, keep attention sinks, a recent window, and important prompt
   rows in the hot tier. Retain additional prompt rows in the bounded cold
   tier; discard rows that fit neither tier. Cold prefill occupancy is capped
   by `sttram_bifurcation_frac`, reserving the remaining space for later
   demotions. This is a capacity/recall/write tradeoff, not a proven universal
   optimum.
2. Expose a bounded number of live cold rows alongside hot rows for real model
   attention. The ordinary policy selects cold rows by recency. Shared Hugging
   Face mask geometry requires a common visible row count across layers;
   identities can differ. A layer with few eligible cold rows can limit all
   layers' exposure.
3. Use observed attention on exposed rows to select promotion candidates.
   Promotion is bounded and re-entry hysteresis prevents immediate repeated
   movement. Unexposed cold rows do not receive direct attention feedback.
4. Append newly generated KV, update score/age metadata, and evict eligible hot
   rows. A valid inclusive cold backup allows re-demotion without rewriting
   that KV payload. When cold capacity is exhausted, reclaim backups or drop
   rows according to the implemented policy; a permanently dropped row cannot
   be retrieved later.

Prompt importance has both a historical key-as-query proxy path and an actual
prompt-query capture path. Their results are different source/configuration
epochs and must not be mixed. Real-query capture and other optional mechanisms
must be explicitly named in each experiment's frozen plan. The ordinary model
policy does not run a query search over every cold key on every decode step.

The standalone `TieredKVCache.step()` simulator also implements page-bound
selection and its own attention routine. That conceptual simulator path must
not be described as the mechanism executed by the ordinary model harness.
The model harness supports batch size one; throughput, concurrency, other
models, and other precision regimes require their own validation.

## Storage, costs, and timing

In the simulated-tier model evaluation, hot and cold tensors reside on the
model device. The cold tier represents STT-RAM through event/byte accounting;
it is not deployed STT-RAM. A separate CPU-backed implementation uses actual
pinned host DRAM and transfers. Its correctness and systems measurements do
not automatically transfer a quality result from the GPU-resident simulation.

KV dtype is a configuration choice. Historical defaults can store FP32 KV;
BF16 model weights alone do not prove BF16 retained KV. Score metadata remains
FP32. Report actual backing storage, metadata, temporary buffers, index state,
and generation-dependent capacity separately from nominal token budgets.
Matching hot capacity or attended width does not establish equal whole-memory
usage. A growing-decode comparator needs its complete resource trajectory.

`src/cost_model.py` derives latency and energy from assumed bandwidth and
per-bit prices. These are scenario outputs, not measured model latency,
whole-system power, or physical STT endurance. Standalone page-bound accounting
assumes ideal persistent bounds and excludes index construction/maintenance
unless explicitly included. Measured model-loop timing requires synchronized
GPU measurements, correct supported backends, identical generated work,
warm-up, repeated blocks, and interference checks.

Decode write avoidance is `saved / (saved + paid)` over potential decode
demotions. It excludes initial cold prompt placement. Whole-request logical
payload accounting must include that placement and identify its counterfactual
policy. Neither ratio measures device page writes, write amplification, or
device lifetime.

## Evaluation and continuation rules

Use the frozen experiment plans and their raw records as evidence. Pair model
weights/tokenizer, exact tokenized input, generation/stopping rules, scorer,
data identities, precision, and declared resource constraints. QA F1 and
GovReport ROUGE-L are different task scores; a task-balanced average must be
labeled accordingly. A full-cache arm is a reference, not a guaranteed quality
upper bound for every generated answer.

The earlier in-harness H2O/SnapKV/StreamingLLM policies are adaptations. They
are functional controls, not evidence of superiority over validated upstream
methods. Official comparator integrations need source pins and numerical/model
fidelity checks; compatibility or correctness changes must be disclosed.
Systems comparisons also require matched runtime controls and fastest correct
supported backends.

Before a new candidate evaluation, freeze the mechanism, control, source/model/
data identities, metric, stopping rule, resource limits, and regression guards.
Develop only on declared development data. Consumed final records remain
consumed; failed candidates stay rejected unless a substantively new,
prospectively registered hypothesis justifies new work. Preserve failed runs
and negative comparisons. Never treat a partial run, passing unit tests, an
internal controller speedup, or an adapted-control win as publication-level
SOTA evidence.

Historical checklists, explanations, and provisional manuscript sources are
preserved byte-for-byte in [the archive](archive/manifest.json). They contain
superseded results and instructions and are not active checkpoints. Extract
them into a separate directory if needed; do not overwrite current entrypoints.
Current audit/results/figure gates must be completed before paper writing.
