# TieredKV project status and next steps

**Current checkpoint — early shared prefill (3 October):** 501 tests pass. Tiny GPU full-logit/KV parity and partial-failure cleanup pass; full-model4K/12 and16K/50 preserve predictions, state, counters and native transfers with all32 early roots and no fallback. Whole-prefill peak falls11.22% to16.5376GB; strict decode GPU active falls5.47%, scoped active including owned pinned/CPU is378.60MiB. Faithful Snap prefill ratio0.9840 passes. [Resource proof](results/hardware/early_shared_cpu_phase_v1/verified_resources.json). Initial resident KV matches341MiB at hot649/cold1982; candidate remains341MiB while Snap grows341–366MiB. Full-trajectory equality remains unresolved. Faithful-baseline quiet3ABBA timing **fails**: CPU54.59 vs Snap25.25ms/token (2.162x), p95CPU59.94 vs Snap25.47ms. TTFT+7.73% and prefill guards pass; median and p95 fail. All300 measured forwards/arm reproduce frozen predictions/state; CPU counters/native transfers and25 quiet observations pass. [Timing proof](results/hardware/early_shared_cpu_phase_v1/verified_cost.json), [figures](results/hardware/early_shared_cpu_phase_v1/figures/manifest.json). Candidate expansion/adoption stops under the preregistered gate; no favorable retry or final-holdout tuning. All owned jobs are terminal. Physical controller measurements, independent quality, generality and final delivery remain open. Repeated10–20day estimate withdrawn; full goal incomplete.

**CPU phase boundary and score-fallback checkpoint (3 October):** **498 tests pass**. Two reproduced regressions exposed a shared batched-score crash on multi-token attention. A single-query shape guard restores the ordinary scorer fallback. Real GPU appends of2/32 tokens preserve exact logits, KV data, metadata and slot order across7 boundaries, then return to native single-token phases. Transformers-owned lazy capture hooks are initialized before ownership checks; both failed harness attempts are preserved. [Fallback proof](results/hardware/cpu_phase_append_fallback_v1/verified_fallback.json).

Three full-model4K/50 checks preserve all predictions/state/counters/native KV: exclusive promotion24832 rows, floor-above-capacity protection, and50 write-aware fallback steps. Native cold reclaim executes1152 times. [Boundary proof](results/hardware/cpu_phase_boundaries_v1/verified_boundaries.json). Fresh current-source4K/12 parity also passes; current-source16K/50 resource and geometry proofs pass, scoped memory433.33 MiB. Fresh current-source3ABBA speed passes: **58.76 to54.12 ms median,7.89% gain/4.64 ms saved**, p95 improves8.87%, TTFT+1.37%. All300 measured predictions/arm, state/counters/native KV and25 quiet observations match. All owned jobs are terminal. The multi-query guard preserves existing ordinary scoring fallback; it does not redesign multi-query attention accumulation. [Current proof](results/hardware/cpu_phase_score_guard_v1/verified_status.json), [figures](results/hardware/cpu_phase_score_guard_v1/figures/manifest.json). Historical speed evidence below remains tied to its archived source.

The ownership audit confirms resident KV roots272.25 MiB constant across51 candidate boundaries versus faithful SnapKV487–512 MiB, growing0.5 MiB/step. Attention width1024 differs on49/50 steps. No fixed SnapKV prompt capacity can equal that constant allocation vector throughout decode. This is an unresolved comparison-design requirement, not permission to claim equality or pad allocations. [Geometry proof](results/hardware/cpu_phase_geometry_v1/verified_geometry.json). Full P0/P1/P2 remains open; no new day estimate.

**CPU phase controller integration (3 October):** **496 tests pass**, including seven integration checks. Native canonical CPU promotion, append and demotion are integrated with pooled GPU metadata and native KV transfers. Recorded 4K/12 matches all predictions, 13 state/visibility snapshots, counters and native KV traffic. All 384 layer transitions use the new CPU phases, with zero fallback. [Correctness proof](results/hardware/cpu_phase_controller_v1/verified_correctness.json).

Fresh 16K/50 matches frozen output/state/counters/native KV and passes scoped memory: **433.33 MiB including owned pinned and CPU array roots**, versus ordinary541.08 MiB. Prefill peak is unchanged. Quiet preregistered3ABBA speed gate **passes: 58.93 to54.09 ms median (8.22%,4.84 ms saved)**, p95 improves8.76%, TTFT+1.40%. All300 measured predictions/arm and frozen state/counters/native KV match;25 GPU quiet observations pass. Longer-context8K/31.5K teacher parity and forced cold256-row slot-recycling parity pass, each50 steps with native CPU phases and no fallback. Free-generation parity passes on all12 reused development documents across six tasks: exact tokens, scores, state, trajectory, counters and native KV. **597 native CPU phase steps/19104 layer transitions** execute with no fallback;174 additional native cold reclaims occur. All owned jobs are terminal. Joint faithful-baseline/quality gates remain open. Candidate remains research-only. Production defaults are unchanged. Full P0/P1/P2 remains incomplete; the unsupported day estimate is withdrawn. This checkpoint supersedes earlier statements that controller integration is unimplemented.

**Native CPU state phase checkpoint (3 October):** **489 tests pass**, including
30 focused phase checks. Fixed shared CPU owners hold hot/cold positions,
grace, shadow/resident flags, streaks and logical/free slot order. Native batch
operations perform promotion, hot append, complete demotion metadata updates
and cold compaction/reclaim. All malformed batch inputs reject before any
layer mutates; version and owner checks reject stale/foreign GPU commands.

All **1600 recorded layer transitions** replay exactly in50 native batches,
including2560 paid rows,23872 restored backups, F32 metadata commands and
physical slot order. Five GPU handoff fixtures across20 layers match original
cache payload/metadata bits, roots and padding: shadow-first reclaim, stable
LRU overflow, live-floor cover, exclusive promotion and eight-layer compaction.
Victim/cold-keep decisions are supplied; these fixtures do not independently
prove retention selection. GPU slot tensors emulate storage, not native host
transfers. Declared32-layer CPU owners are **1116896 bytes (1.065 MiB)**;
container/output/Python workspace and whole-model peaks remain unmeasured.

[Phase proof](results/hardware/cpu_state_transition_v1/verified_stage.json).
**Decode-controller integration remains incomplete.** Next attach canonical
CPU owners and consume these commands through promotion, demotion, native KV
transfers, visibility and cold reclaim, preserving unsupported fallback; then
run recorded model parity, actual resource and preregistered speed gates.
No speed/quality/adoption claim is established. All owned jobs are terminal;
production defaults and failed-candidate stops remain unchanged. Full P0/P1/P2
remains incomplete; day estimate withdrawn.

**CPU transition diagnostic and corpus (3 October):** Both fresh policy
profiles preserve all50 frozen predictions, final state, counters and native KV.
The0.836-second first extension load is already excluded by benchmark warmup;
it does not reverse the pooled controller's failed speed decision. Native
backup classification costs3.009 inclusive instrumented ms/step across1600
calls. Isolated backup batching is not justified as a4ms whole-loop fix.
Broader CPU state-transition work is the next mechanism, not another wrapper.

A recorded ordinary GPU run freezes **1600 layer transitions**. Independent
CPU/F32/slot replay is exact:2560 paid rows,23872 restored backups and1600
queued compactions. Positions, flags/streaks, metadata bits, logical/free slots
and outcomes match. The native backup classifier matches every record. This
corpus supports CPU-stage validation without repeated model launches per edit.
Cold occupancy reaches897/1090; no overflow occurs. Saturation/reclaim and
broader malformed/ragged/nonfinite fixtures remain mandatory.

[Diagnostic](results/hardware/pooled_controller_diagnostic_v1/verified_diagnostic.json),
[corpus](results/hardware/controller_transition_corpus_v1/verified_corpus.json),
[next phase plan](results/hardware/cpu_state_transition_v1/plan.json).
**Full CPU state-transition redesign is not implemented yet.** Existing459
unit tests remain passing; this checkpoint adds executable replay evidence,
not a speed/quality/adoption claim. All owned jobs are terminal; wrappers,
hooks and pinned roots release. Production defaults and failed-candidate
stops remain unchanged. Full P0/P1/P2 remains incomplete; day estimate withdrawn.

**Pooled controller integration checkpoint (3 October):** **459 tests pass**.
The research controller now uses fixed GPU metadata owners, fused age/tail
preparation and bounded KV/F32 compaction. Recorded 4K/12 preserves all
predictions, 13 visibility/state frames, counters and native KV traffic;
12 pooled preparations and 48 compaction groups execute with no fallback.
Fresh 16K/50 also matches frozen predictions/state/counters/native KV.
Scoped active memory is **432.10 MiB** versus ordinary541.08 MiB (20.14% less);
with conservative CPU array roots it is432.27 MiB, below512 MiB. The417792-byte
GPU arena and descriptor work are charged by GPU peaks. Pooled descriptors
upload6457600 logical bytes per50-step trial, separately from KV traffic and
5555328 timestamp-rank descriptor bytes. Prefill peak is unchanged.

[Correctness proof](results/hardware/pooled_controller_v1/verified_correctness.json),
[resource proof](results/hardware/pooled_controller_v1/verified_resource.json).
Quiet3ABBA is complete: median **58.06 to61.54 ms (6.00% slower,3.48 ms
added)**. Both speed primaries fail; p95 +0.67% and TTFT +0.87% pass. All300
measured forwards per arm replay frozen predictions/state/counters/native KV;
all25 quietGPU0 observations pass. **Stop pooled controller expansion/adoption.**
No favorable timing retry or wider quality/baseline expansion for this candidate.
[Timing decision](results/hardware/pooled_controller_v1/verified_cost.json).
All owned jobs are terminal and hooks/leases/pinned roots release. Production
defaults are unchanged. Full P0/P1/P2 remains incomplete; day estimate is
withdrawn. Primitive proof remains historical component evidence, not full
model proof. Two failed startup attempts are retained, including the corrected
generated-helper name collision; neither measured candidate timing.

**Native controller prototype (3 October):** **437 tests pass**, including
36 focused native core/integration checks. A native CPU module, built with
existing Torch headers and system g++, computes cross-layer promotion plans,
hysteresis/live-floor transitions, backup classification and read addresses.
The research policy also groups timestamp ranking by identical live length,
using the verified fused gather and original unstable CUDA argsort. Grouped
sorting matches separate GPU sorting in 44 fixtures, including ties/nonfinite
values. No process-global patch, retention change or new dependency is used.

The recorded 4K/12-step GPU pair preserves all predictions, 13 cache/visibility
frames, counters and native KV transfers. There are 12 native plans, 384 backup
classifications, 25 address calculations and no planning fallback. Fresh
16K/50 matches all predictions, final state, counters and native KV traffic.
Scoped active memory is **432.12 MiB** versus ordinary 541.08 MiB (20.14% less),
passing the 512 MiB ceiling; with conservative CPU array roots it is432.29 MiB.
Prefill peak is unchanged; C++/Python allocator internals remain unmeasured.
Timestamp rank descriptors upload **5.555 MB** per measured trial; bounded
ownership descriptors derive12.864 MB. These are separate from native KV and
are charged by timing/GPU peaks. Quiet3ABBA is complete: median **57.92 to 62.80 ms (8.43% slower, 4.88 ms
added)**; p95 **64.38 to 66.79 ms (3.74% worse)**. TTFT +0.25% passes. All
300 measured forwards per arm replay frozen predictions, final state, counters
and native KV traffic; all25 quietGPU0 observations pass. **Stop native
controller expansion/adoption:** both speed primaries and p95 fail. No favorable
timing retry or wider quality/baseline expansion for this implementation.
[Timing decision](results/hardware/native_controller_v1/verified_cost.json).
All owned jobs are terminal; hooks, leases, methods and pinned roots release.

The native core proves promotion, backup and address calculations, not the full
controller gate. Promotion/address/ranking are grouped; backup classification
still enters native code once per layer. Complete transition pooling remains
unimplemented. Combining wrappers around the existing layer state has failed
the joint gate; the next mechanism must redesign the complete hot/cold state
layout and account for retention/quality, rather than expand these wrappers. Descriptor uploads and CPU arrays
must be reported separately from native KV transfers. Failed ownership-only
candidates remain stopped; production defaults are unchanged. Full P0/P1/P2
remains incomplete; the unsupported day estimate stays withdrawn.

**Bounded deferred ownership checkpoint (3 October):** **401 tests pass**,
including 16 focused checks. A distinct research transition defers compaction
when cold reclamation is unnecessary; overflow preserves immediate compaction.
Two passes of the existing verified native kernel gather to separate scratch
and copy back safely. Groups contain at most eight layers. GPU 4K/12 matches
all 12 predictions, 13 cache/visibility frames, counters and native KV traffic.
There are 384 deferred compactions in 48 groups, no immediate compactions or
append fallbacks, and scratch is bounded at **30.97 MiB**. Fresh 16K/50 exactly replays frozen predictions, final state, counters and
transfers. Scoped active memory is **432.12 MiB**, **20.14% lower** than ordinary
541.08 MiB, passing both the 512 MiB ceiling and 10% reduction gate. Prefill
peak is unchanged. Quiet 3ABBA is complete: median **58.33 to 63.09 ms (8.15% slower)**,
p95 **64.77 to 70.88 ms (9.43% worse)**; TTFT +0.36% passes. All 300 measured
forwards per arm exactly replay predictions, state, counters and native KV
traffic; 25 quiet GPU0 process checks pass. **Stop bounded deferred ownership
expansion/adoption:** speed primaries and p95 fail. No favorable timing retry.
[Timing decision](results/hardware/bounded_deferred_ownership_v1/verified_cost.json).

Reporting correction: the copied timing verifier's zero extra metadata scalar
is false. The native wrapper uploads descriptors: fixed geometry/source derives
**12.864 MB and 1200 upload calls per measured candidate trial**. These are
separate from native KV manager bytes/calls; total extra versus ordinary was
not independently instrumented. Timing and memory charge the descriptor work.
The original artifact remains retained and the failed decision is unchanged.
[Correction](results/hardware/bounded_deferred_ownership_v1/verified_descriptor_reporting_v2.json).
All owned jobs are terminal; hooks, instance methods and pinned roots release. Failed predecessors remain
stopped. [Correctness](results/hardware/bounded_deferred_ownership_v1/verified_correctness.json).
Next target is broader controller consolidation; bounded grouping alone has
failed to deliver a joint win. Do not expand this failed implementation. Production defaults are
unchanged; full P0/P1/P2 remains incomplete and the day estimate is withdrawn.

**Current regression diagnostic (3 October):** fresh paired GPU0 policy profiles
preserve all 50 predictions, final state, counters and native transfers in each
arm. Shared ownership performs **1600 per-layer in-place gathers**; ordinary
bulk performs **50 batched native gathers**. Instrumented compaction is
6.76 ms/decode versus 3.12 ms for ordinary compaction preparation; ordinary
native gathering adds 0.75 ms in its visibility boundary. Shared reservation
and hot extension also add validation/copy work. These overlapping profile
ranges are localization evidence, not a causal savings estimate or timing gate.

The code trace establishes the ordering constraint: after demotion, cold
reclaim requires compacted hot positions and scores; it does not read compacted
hot KV. A distinct ownership implementation could defer payload gathering with
bounded, alias-safe scratch. However, the ordinary controller still spends
instrumented time in preparation (10.96 ms), eviction (11.75 ms) and visibility
(9.86 ms); these ranges overlap and cannot be added as savings. Nearly one
million calls per 50 steps remain. Removing this regression alone would not
prove the required speed win or close the faithful-baseline gap.
[Verified diagnostic](results/hardware/shared_transition_diagnostic_v1/verified_diagnostic.json).

Next target is broader controller consolidation, with a preregistered complete
mechanism and actual whole-loop/resource/quality gates. Do not repeat the failed
shared implementation or promote a profile as a benchmark. All owned jobs are
terminal; production defaults are unchanged. Full P0/P1/P2 remains incomplete;
the unsupported day estimate remains withdrawn.

**Shared ownership milestone (3 October):** **385 tests pass**, including 11 focused
unified-transition checks. The GPU 4K/12-step pair matches all predictions,
13 cache/visibility-state frames, counters and native KV traffic. Fresh 16K/50
matches the frozen ordinary predictions, final state, counters and transfers.
Unified model append, promotion and immediate compaction share consumed buffers.
Scoped active memory falls **541.08 to 415.16 MiB (23.27%)**, passing both the
512 MiB ceiling and 10% reduction gate. This includes the owned 256 MiB pinned
allocator block; it excludes constant model storage and unowned process memory.
Whole reserved envelope remains 5089.49 MiB; prefill peak remains 18.62783 GB.
[Resource proof](results/hardware/shared_decode_transition_v1/verified_resource.json).

The standalone append reservation was stopped at **531.00 MiB**, below neither
512 MiB nor a 10% reduction. Its failure is preserved separately.
[Standalone stop](results/hardware/reserved_decode_append_v1/verified_resource.json).
Quiet 3ABBA timing is complete: median decode **58.84 to 64.13 ms (8.98%
slower, 5.28 ms added)**; p95 **64.55 to 69.32 ms (7.38% worse)**. TTFT +0.78%
passes. All 300 measured forwards per arm replay predictions, final state,
counters and native KV transfers; 25 quiet GPU0 observations pass. **Stop
unified-transition expansion/adoption:** both speed primaries and the p95 guard
fail. No favorable timing retry or wider quality/baseline run for this failed
candidate. [Timing proof](results/hardware/shared_decode_transition_v1/verified_cost.json),
[measured tradeoff](results/hardware/shared_decode_transition_v1/ownership_tradeoff.pdf).
All owned jobs are terminal; leases, hooks and pinned roots release.

The memory obstacle is resolved only for this experimental design and measured
scope. The joint memory/latency/quality acceptance gate remains open. Next
localize the added cost of consumed-buffer compaction before proposing another
mechanism; do not widen this failed implementation. Independent quality, current
faithful baselines and physical hardware evidence remain open. Full P0/P1/P2
is incomplete; the unsupported 10–20-day forecast remains withdrawn.

**Current CPU timestamp decision (3 October):** **358 tests pass**, including
11 focused checks. Cold timestamp transitions move toCPU; oneGPU snapshot
pervisibility boundary preserves the originalGPU unstabletie sorting.
GPU4K12 matches12predictions,13cache/visibility-state frames, counters and
nativeKVbytes/calls; eightGPU tie-order cases pass. Native kernels, ranking,
retention, cache storage and ordinary source/CLI/defaults are unchanged.
[Correctness](results/hardware/cpu_lru_metadata_v1/verified_correctness.json).

Fresh16K50 replays all frozenpred/state/counters/nativeKVpayload/calls.
Extra timestamp H2D **5.692MB** / initialD2H **0.1046MB** are charged separately;
newCPU timestamp+temporarypacked tensor peak is229632bytes. Scoped active
memory becomes **541.19MiB** versus541.08MiB ordinary; reserved5089.71MiB and
prefill18.62783GB. This passes576MiB pilot, **fails512MiB**; CPUmallocrounding/
otherhostretention remains unmeasured.
[Resource proof](results/hardware/cpu_lru_metadata_v1/verified_resource.json).

Quiet3ABBA whole-loop median **57.85→57.34ms (0.88% faster,0.51ms saved)**;
p95 **64.46→61.51ms**, TTFT+0.09% pass. All300 measuredforwards/arm replay
pred/state/counters/nativeKVtraffic/calls and candidate timestampcopies;
25 before/after quietGPU0 checks pass. **Both >=5% and >=4ms primaries fail.
Stop standalone CPUtimestamp expansion/adoption**, with no favorabletiming
retry or widergeneration/quality/faithfulbaseline run for this candidate.
[Timing/stop proof](results/hardware/cpu_lru_metadata_v1/verified_cost.json),
[figure](results/hardware/cpu_lru_metadata_v1/timestamp_tradeoff.pdf).

All owned jobs are terminal; temporarymirrors and pinned roots release.
The failed projection/timestamp-only screens leave broader hot/cold transition
and ownership restructuring as the target; require a preregistered complete
mechanism with whole-loop/resources and eventualfaithfulbaseline/quality gates.
FullP0/P1/P2 remains open; the unsupported10–20-day forecast stays withdrawn.

**Current ordinary-backend screen (3 October):** **347 tests pass**, including
19 focused projectedscore/victim cases. The research coordinator prepares
victim score/age projections once across layers rather than creating temporary
CUDA tensors per layer. SixGPU cases preserve projection bits/victims for
ragged/nonfinite/grace inputs and1/4/32-token appends. Original ranking, stable
ties, protection, promotion, cache types and unsupported fallback stay intact.

Charged complete preparation+selection 3ABBA yields **3.419→2.686ms**,
only **0.733ms saved** against the >=4ms gate. Candidatep95 **5.831ms** versus
control3.502ms (**1.665×**) fails its guard; all three medians improve and
25 before/after quietGPU0 checks pass. **Stop this projection-only candidate**
before fullmodel/resource/timing/quality expansion. CPU/GPU stage equality
is not proof of fullmodel candidate preservation or a mainlatency-gate win.
Ordinary source/CLI/defaults are unchanged; all owned jobs are terminal.
[Verified screen](results/hardware/batched_eviction_projection_v1/verified_stage.json).

The screen localizes preparation's limits: the broader eviction/demotion,
layer-state transitions and synchronization remain the structural target.
Next preregister a complete state-transition mechanism; do not retry this
failed stage or widen failed fullquery candidates. FullP0/P1/P2 remains open;
the unsupported10–20-day forecast stays withdrawn.

**Current critical-path diagnostic:** the GPU0 ordinary-bulk16K/50-step
profile replays every frozen prediction, finalcache, logicalcounter and native
payload/call total. All owned jobs are terminal; hooks and pinned roots release.
About967,000 policy calls take1.920s under instrumentation. Eviction/demotion
is11.48ms/decode; read preparation11.10ms and visibility9.94ms are overlapping
cumulative ranges, not additive speed estimates. Native visible gathering is
0.76ms/decode. Next preregister cross-layer eviction/demotion bookkeeping
consolidation on the ordinary backend and require preserved state/selection/
transfers plus whole-loop/resource gates. Fullquery derivatives remain stopped.
[Verified critical path](results/hardware/completion_critical_path_v1/verified_audit.json).

**Completion plan correction (3 October):** the repeated **10–20 days** estimate
is withdrawn; it was not recalculated from the remaining gates. Earlier
estimates below are historical and must not be treated as a current forecast.
[Critical-path audit](results/hardware/completion_critical_path_v1/audit.json)
separates failed research gates from incomplete deliverables and external hardware.
The existing ordinary CPU backend is541.08MiB versus528.74MiB SnapKV; both
exceed512MiB and attention widths differ on49/50steps. A frozen older faithful
latency study is60.56 versus25.83ms (2.34×); its independent quality comparison
also fails NI. These studies do not prove current bulk quality or a joint win.
PhysicalSTT/controller energy/endurance remains unavailable and unproven.

Execution priority: profile the current ordinary bulk/lastlogits backend with
frozen50-step state/prediction/payload replay; address one structural bottleneck
that can close a main memory/latency gate. Keep failed fullquery derivatives
stopped. Then verify actual resources before expanding independentquality and
baselines; prepare reproducibility/manuscript artifacts alongside those gates.
No full completion date is supported until the research and hardware gates resolve.

**Current channel-reader decision (3 October):** **328 tests pass**. An exact
native reader scans existing cold keys and fetches only selected cold values;
query/ranking math is unchanged. GPU4K/12-step pairs replay all predictions,
13 logit/cache/counter frames,384 selections and both arms' destination/prefix
ownership. Native BF16/FP16/F32 and36 metadata guards, stream/pool/root cleanup
pass. [Correctness](results/hardware/query_channel_reads_v1/verified_correctness.json).

Fresh16K/50-step replay verifies **45.53% less copied K/V payload**
(**5.826→3.173GB**), unchanged D2H115.86MB and index copies75.33MB. NativeH2D
calls increase1701→3301. Active decode remains **545.17MiB**, above512MiB but
within the576MiB pilot; reserved5089.49MiB, prefill18.62783GB and roundedhost
256MiB are unchanged. This measures copied nativeKV payload, not all library
transfers or physicalPCIe/STT reads.
[Resource proof](results/hardware/query_channel_reads_v1/verified_resource.json).

Quiet3ABBA component comparison yields **97.48→93.02ms (4.57% faster,
4.46ms saved)**; p95100.49→95.90ms and TTFT+0.05% pass. All300 measured
forwards/arm replay frozenpred/state/counters and each arm's payload/calls;
25 quiet-process checks pass. **The >=5% primary fails. Stop standalone
channel-reader expansion/adoption.** No fresh ordinaryLRU or independentquality
expansion follows this failed gate; no favorable timing retry.
[Timing/stop proof](results/hardware/query_channel_reads_v1/verified_cost.json),
[figure](results/hardware/query_channel_reads_v1/channel_tradeoff.pdf).

Ordinary source/CLI/defaults are unchanged by this mechanism; all owned jobs
are terminal and own hooks/index metadata/pinned roots release. Full512MiB,
matched baselines, independentquality, physicalSTT and P0/P1/P2 remain open.
Next preregister a distinct mechanism targeting the remaining query-loop or
synchronization cost, then require correctness/resources/component/LRU gates.
Estimate **10–20 working days**, conditional on hardware/research outcomes.
Earlier checkpoints below are historical evidence, not current next actions.

**Current visible-buffer decision (3 October):** **306 tests pass**. The
research selector retains the existing visible hot prefix and writes only
selected cold rows. GPU4K/12-step pairs exactly replay predictions, all logit/
cache/counter frames,384 visible-K/V selections and transfer metadata;384
candidate destinations retain their identity, offset and strides. Canonical
hot K/V remain unchanged and own hooks/index metadata release.
[Correctness](results/hardware/query_visible_suffix_v1/verified_correctness.json).

Fresh16K/50-step resource replay passes the576MiB pilot at **545.17MiB**,
versus547.49MiB query concat. This still exceeds512MiB; reserved5089.49MiB,
prefill18.62783GB and roundedhost256MiB are unchanged. NativeKV H2D remains
5.826GB, plus75.33MB of separately charged index copies. Three omitted native
source files were added in a SHA-matched supplemental archive; original source
archive and verifier are preserved. [Resource proof](results/hardware/query_visible_suffix_v1/verified_resource_v2.json).

Quiet3ABBA component comparison passes: **105.32→97.05ms (7.85% faster,
8.27ms saved)**, p95108.21→99.88ms; TTFT+0.34% passes. Frozen states, predictions,
counters, native bytes/calls and metadata replay;25 quiet-process checks pass.
But fresh ordinaryLRU comparison fails: **57.87→97.56ms (1.686×)**,
p9564.51→100.59ms (1.559×); TTFT+0.46% passes. **Stop standalone suffix-query
expansion/adoption.** The component improvement remains research-only.
[Component proof](results/hardware/query_visible_suffix_v1/verified_cost.json),
[LRU stop decision](results/hardware/query_visible_suffix_v1/verified_lru.json),
[comparison figure](results/hardware/query_visible_suffix_v1/suffix_tradeoff.pdf).

Next preregister a distinct lower-traffic reader: keys for ranking, values only
for selected rows, preserving existing Q/ranking/selection math. Ordinary source/
CLI/defaults are unchanged; all four owned GPU jobs are terminal. Full512MiB,
actual matched baselines, independent quality, physicalSTT and P0/P1/P2 remain
open. Estimate **10–20 working days**, conditional on hardware/research outcomes.

**Current diagnostic decision (3 October):** **287 tests pass**. The saved
16K/50-step GPU-query diagnostic replays the frozen prediction/state/counter/
transfer assertions. A corrected summary verifies **eight CPU phases, 1,600
observations each**, excluding six duplicate GPU annotation spans. All owned
jobs are terminal and own hooks/index metadata are released.

Visible-K/V composition is the largest instrumented CPU range: **15.89 ms per
decode**, including launch/waiting/profiler overhead. This is localization,
not a speedup estimate. Chrome export did not run; native GPU-kernel attribution
is incomplete. Failed reports and source archives are preserved.
[Diagnostic evidence](results/hardware/query_phase_profile_v1/verified_summary.json).

Next: preregister visible-buffer reuse with cold-row-only updates, then prove
ownership/parity/resources before component and ordinary-LRU latency gates.
The standalone fullscan candidate stays stopped at **1.810× median / 1.679×
p95** despite its **24.61%** read-index component improvement. Full512MiB,
actual matched baselines, independent quality, physical STT and P0/P1/P2 remain
open. Estimate **10–20 working days**, conditional on hardware/research outcomes.

**Current GPU read-index decision (3 October):** a research batchedGPU source
map uses the existing nativeH2D kernel while keeping full-query selection math
unchanged. TenCPU address-map/invalid-range tests and **283 tests** pass.
NativeBF16/FP16/F32 exact reads, stale/unconsumed guards, serialstream caller
handoff, alternatingpool registrations and pinnedroot/metadata cleanup pass.
GPU4K12 pairs exactly replay12predictions/13logit-cache-counterframes/384KV
choices and unchanged nativeKVbytes; no canonicalCPUreads.
[Correctness proof](results/hardware/gpu_query_read_indices_v1/verified_correctness.json).

Fresh16K50 preserves full-query predictions/state/counters/KVbytes and passes
its576MiB active-envelope pilot at **547.488MiB**, versus544.321MiB oldquery.
Roundedownedhost256MiB/reserved5089.492MiB/prefill18.62783GB unchanged.
NativeKVH2D remains **5.826GB**; candidate index-metadataH2D **75.33MB** and
GPUindex storage1.409MiB are separately charged. The old generic-index metadata
copies were not separately measured, so this supplies no totalH2Dsaving claim.
Ownroots/hooks/indexmetadata release. The original resource proof's copied
next-gate label is preserved and corrected in a new proof: **component cost
only, no development-quality expansion**.
[Corrected resource proof](results/hardware/gpu_query_read_indices_v1/verified_resource_v2.json).

Quiet3ABBA originalquery-versusGPUindex-query component cost passes:
median **139.451→105.133ms (24.61% faster)**, p95 **142.705→108.065ms**, TTFT
**2192.839→2197.709ms (+0.22%)**. Allfrozenquery states/teacherpredictions/KVbytes
replay,25GPU0computePID checks quiet and ownhooks release.
[Component proof](results/hardware/gpu_query_read_indices_v1/verified_cost.json).
Fresh ordinaryLRU-versuscandidate3ABBA fails the whole-path guard: median
**58.215→105.362ms (1.810×)**, p95 **64.477→108.280ms (1.679×)**; TTFT
**2219.275→2227.665ms (+0.378%)** passes. Allfrozenarm states/predictions/KVbytes
replay,300measured forwards/arm and25quietGPU0 process checks verified.
**Stop standalone full-query GPU-index candidate expansion/adoption.** The
component speedup remains useful research evidence and cannot override this
guard. Next inspect fused ranking and lower-traffic retrieval with GPU-generated
selected-row indices; the fixed8-page heuristic stays stopped.
[Whole-path stop decision](results/hardware/gpu_query_read_indices_v1/verified_lru.json).
[Diagnostic figure](results/hardware/gpu_query_read_indices_v1/read_index_tradeoff.pdf). Ordinary source/CLI/defaults remain
unchanged. Full512MiB/actual matching, independentquality, physicalSTT and full
P0/P1/P2 remain open. Estimate **10–20 working days**, conditional on hardware/
research outcomes.

**Projection-reuse decision (3 October):** the research selector runs
after the actual nativeMistral query projection and before nativeKV update.
Only the duplicate projection expression changes; rotary/softmax/ranking/
selection math stays the same. SevenCPU preservation/partial-error/constructor
cleanup tests pass; **273 tests pass** overall. FullGPU4K12 nativeCPU pairs
match every prediction/logit/cache/logicalcounter frame, visibleKV/choice and
H2D/D2H byte, replaying the frozen full-query control. Projection calls drop
**768→384**, canonicalCPUreads0, ownhooks and pending contexts clear.
[Verified correctness](results/hardware/reused_query_projection_v1/verified_correctness.json).

Quiet3ABBA visible-only cost compares the original duplicated-Q hook with
native-projection reuse, zeroextra coldreads and exact ordinarystate/transfers.
Median **82.129→81.559ms**, saving **0.570ms /0.694%**, fails the predeclared
at-least5%primary. p95 **87.894→87.351ms** and TTFT **2192.039→2194.349ms** guards
pass. All300measured teacher forwards/arm replay,25GPU0 computePID checks
quiet, ownhooks removed. **Stop standalone projection-reuse expansion.**
Correct reuse remains research knowledge; no ordinary adoption or latency-win
claim. [Verified cost decision](results/hardware/reused_query_projection_v1/verified_cost.json).
Next inspect fused ranking and GPU-generated native read indices. This does not reopen the stopped full-scan
quality/timing/adoption path or fixed8-page heuristic. Ordinary source/CLI/
defaults are unchanged; full512MiB/matchedquality/latency, physicalSTT and full
P0/P1/P2 remain open. Estimate **10–20 working days**, conditional on hardware/
research outcomes.

**Query-cost diagnosis (3 October):** three quietGPU0 ABBA blocks
compare ordinaryBulkLRU against the existing unfused query-math hook pattern
using only already-visible32cold rows. No extra cold reads occur. All300
measured teacher forwards per arm, final cache state, logical counters and
nativeH2D/D2H bytes exactly replay ordinary control; ownhooks release.
Median decode **58.581→82.366ms (+40.60%)** and p95 **64.355→87.390ms (+35.79%)**
exceed the2%screen; TTFT **2177.814→2192.567ms (+0.68%)** passes. This diagnostic
includes query projection/rotary/ranking/host-ID/visible concatenation and is
not a formal lower bound on other implementations. **Reduced cold transfers
alone cannot rescue this particular unfused hook pattern.**
[Verified diagnosis](results/hardware/query_hook_cost_v1/verified_report.json).

Projection reuse was subsequently verified but failed its standalone cost
primary; its decision appears above. Fused ranking and GPU read indices remain
next research targets. The fixed8-page centroid
screen and full-scan query candidate stay stopped. Ordinary source/CLI/defaults
remain unchanged; **266 tests** remain verified. Full P0/P1/P2 remains open;
estimate **10–20 working days**, conditional on hardware/research outcomes.

**Retrieval-screen decision (3 October):** a read-only fullGPU16K/
50-step native-query audit replays all predictions, final cache state and
logical counters, with ownhooks/canonical guard restored. Fixed eight centroid
pages of16 stable cold rows cover **13.49%** of live rows but retain only
**28.39%** of the full-query top32 and **55.40%** of their attention mass, even
using the best rerank under the full reference distribution. Both predeclared
recall screens fail; no selection underfills. **Stop implementation and wider
quality/timing tests of this fixed8-page heuristic.** This does not reject all
pruning methods. Summaries are rebuilt offline; projected candidate bytes are
not measured H2D savings or maintained-index cost. The existing index also
reads canonical CPU tensors and cannot directly serve the native pool.
[Verified screen](results/hardware/native_page_recall_v1/verified_report.json).
[Per-layer diagnostic](results/hardware/native_page_recall_v1/recall_screen.pdf).

The later query-hook cost diagnosis appears above; this fixed8-page
heuristic remains stopped. Ordinary source/CLI/defaults
remain unchanged and **266 tests** remain verified. Full P0/P1/P2 remains open;
estimate **10–20 working days**, conditional on hardware/research outcomes.

**Full-scan query-selector decision (3 October):** the research native-pool
current-query selector passes four tiny real-Mistral pairs and full GPU4K/12-step
original-GPU/paired-GPU/native-CPU checks: all predictions,13 cache frames and
384 visible-KV/choice observations match. The initial CPU/CUDA index-device
failure is preserved; a one-line device correction passes the same strict
checks. **266 tests pass.** Ordinary CLIs/defaults are unchanged.

The fresh16K/50-step query resource pilot passes its predeclared576MiB combined
active ceiling at **544.321MiB**, versus541.077MiB ordinaryLRU. Its256MiB rounded
owned host capacity,5089.49MiB reserved envelope and18.6278GB prefill peak remain
unchanged; all50 teacher predictions match and all owned roots/hooks release.
Full-scan current-query ranking charges **5.826GB nativeH2D** versus0.316GB for
LRU. **No speed or independent-quality claim.** The frozen12-request/six-task
paired development check passes with **+2.132 task-balanced points**, descriptive
CI[+0.179,+4.045]. Six prediction pairs change; all12 controls replay16 frozen
integrity fields. This is reused development data, not independent quality.
Quiet3ABBA timing fails the preregistered guards: median decode **58.684→
139.831ms (2.383×)** and p95 **64.720→143.339ms (2.215×)**; TTFT guard passes
2229.506→2211.170ms. All300 measured teacher forwards per arm replay;25
before/after GPU0 compute-process observations are quiet and ownhooks release.
**Stop wider candidate quality/timing expansion and ordinary adoption.**
The score gain is a research tradeoff with full-scan transfer/latency cost.
[Correctness proof](results/hardware/pool_query_selector_v1/verified_correctness.json).
[Resource proof](results/hardware/pool_query_selector_v1/verified_resource.json).
[Development proof](results/hardware/pool_query_selector_v1/verified_development.json).
[Timing stop decision](results/hardware/pool_query_selector_v1/verified_timing.json).
[Diagnostic figure](results/hardware/pool_query_selector_v1/query_tradeoff.pdf).

Full P0/P1/P2 remains open, including complete512MiB/actual-byte-and-width
matching, faithful-baseline quality/latency, physicalSTT/controller, robustness,
model2/serving and final paper proof. Estimate **10–20 working days**, conditional
on hardware access and research outcomes.

**Packed-capacity decision (3 October):** a configuration-only
pilot fills the real256MiB pinned pool with1984 cold slots instead of1090.
Occupied cold KV rises112.125→196MiB (**74.8% more**) without increasing the
physical pinned allocation; active decode rises541.077→541.155MiB. Prefill and
reserved peaks are unchanged. This resource pilot passes, but12 paired
**development** requests/six tasks give a task-balanced score difference
**−0.329 points**, descriptive CI[−1.124,+0.466]. The nonnegative-direction
pilot gate fails. Three generated predictions change and3125 extra initial
cold rows become visible: retention reaches attention but establishes no
quality gain. All12 current-API controls replay frozen ordinary rows in16
fields, and development contexts are disjoint from the100-document final
holdout. **Stop wider quality/timing expansion and adoption.** Both packed-pool
GPU jobs finish and all owned roots/hooks release. Ordinary code/configuration
is unchanged; current ordinary-source validation remains **262 tests**.
[Resource proof](results/hardware/packed_cold_capacity_v1/verified_resource.json).
[Development stop decision](results/hardware/packed_cold_capacity_v1/verified_development.json).

**Verified reserved SnapKV baseline prototype:** genuine1024-row backing
charges512MiB KV capacity at all51 boundaries while exposing only populated
974–1024-row prefixes. Eight tiny CPU pairs, partial-error/overflow checks and
fullGPU paired51-frame checks preserve every populatedKV bit and all logits;
50 predictions replay the frozen original. Its complete decode active scope
is522.61MiB, exceeding512MiB, and reserved envelope rises2881.49→3329.49MiB.
Both owned GPU jobs finish;64 roots and adapter hooks/groups release. Keep
this faithful-selection storage variant research-only; the original growing
adapter remains the baseline control. No speed/memory-win, exact TieredKV-use
match or ordinary baseline replacement claim.
[Verified baseline fidelity/resources](results/hardware/snap_reserved_capacity_v1/verified_report.json).

Full P0/P1/P2 remains open: joint actual-budget quality/latency, complete512MiB
scopes, physicalSTT/controller, robustness/model2/serving and final paper proof.
The later native-pool current-query decision appears above; this packed
capacity variant remains stopped. Estimate **10–20 working days**,
conditional on hardware access and research outcomes.

**Verified actual-resource audit (3 October):** fresh full-model GPU runs
with the adopted last-prefill-logit option reproduce all 50 frozen predictions
for ordinary bulk TieredKV and faithful SnapKV. Direct observations cover
**3200 layer-forward widths**: TieredKV uses1024 rows throughout, while SnapKV
grows975–1024;49/50 steps differ. Measured decode GPU allocation excluding
constant model storage is285.08MiB for TieredKV plus256MiB owned pinned capacity,
versus528.74MiB GPU-only for SnapKV. Combined active envelopes are therefore
**541.08 / 528.74MiB**; both exceed512MiB. Combined reserved envelopes are
**5089.49 / 2881.49MiB**, and whole-prefill peaks **18.6278 / 16.9583GB**.
CPU state/counters/transfers/phase peaks exactly replay its frozen ordinary run;
adapter groups/hooks and all owned roots are released. Both GPU jobs finished.
**Actual bytes and widths are unequal: the matched-resource P0 gate remains open.**
This audit supplies no new timing or independent quality result; the frozen
older DirectCPU NI failure remains separate evidence. Ordinary source is
unchanged and262-test validation remains current. Next preregister a comparison
that controls actual allocation ceilings and observed attention geometry before
quality/latency reruns. Full P0/P1/P2 remains open. Estimate **10–20 working days**,
conditional on hardware access and research outcomes.
[Verified actual-resource comparison](results/hardware/snap_actual_envelope_v1/verified_report.json).
[Diagnostic figure](results/hardware/snap_actual_envelope_v1/resource_gap.pdf).

**Verified layerwise-prefill experiment (3 October):** placing future KV
one layer at a time, while returning full raw KV to current prefill attention,
reduces the measured whole-prefill peak **18.6278 → 16.7095 GB (10.30%)**.
Eight tiny real-Mistral pairs preserve full prefill/decode logits, cache/state
and counters; partial-error reset and hook cleanup pass. The full GPU control
replays frozen ordinary results, and the candidate preserves all 50 predictions,
state, counters and transfer bytes/calls. Both jobs finished and roots/hooks
released. The 10% memory primary passes, but GPU-only decode active allocation
excluding constant model storage increases **285.08 → 291.65 MiB (2.31%)**,
failing the 2% guard. Including the unchanged 256 MiB pinned host capacity gives
**541.08 → 547.65 MiB (1.22%)**; that wider denominator does not establish the
stricter guard. **Research tradeoff only: stop expansion and ordinary adoption.**
Second geometry, GPU generation and timing runs are not launched. Ordinary
source is unchanged; existing **262-test** validation remains current. Next
return to P0 actual-resource matched quality/latency. Full P0/P1/P2 remains open;
estimate **10–20 working days**, conditional on hardware and research outcomes.
[Verified tradeoff and stop decision](results/hardware/layerwise_prefill_placement_v1/verified_guard_review.json).

**Verified GPU allocation diagnosis (3 October):** the unchanged 16K
Mistral prefill reaches **18.6278 GB** at the final MLP multiplication,
while retaining **2048 MiB** of full prompt KV. Each MLP operand/product
is **448 MiB**. Query scoring is not the origin of the first global peak.
Two accepted observers replay all 50 predictions, policy state, counters,
transfers and phase peaks against the frozen uninstrumented ordinary run.
Diagnostic hooks and roots are released; two initial cleanup assertion failures
are preserved and traced to 64 native Transformers output-capture hooks.
The accepted run establishes those native hooks before the strict baseline.
All four owned GPU jobs finished. Ordinary source is unchanged; the existing
**262-test** validation still applies. This is allocation diagnosis, not a
measured optimization. The layerwise experiment above now tests that mechanism
and stops on its decode-memory guard; continue actual-resource comparisons. Actual resource equality, matched
quality/latency, physical STT and full P0/P1/P2 remain open. Estimated remaining
effort: **10–20 working days**, conditional on hardware and research outcomes.
[Verified allocation audit](results/hardware/prefill_allocation_audit_v1/verified_report.json).

**Verified fixed-stream profiler fix:** FullCache, StreamingLLM, H2O and
SnapKV now profile fixed teacher tokens without requiring TieredKV's STT
counter keys. Policy counter deltas are emitted when available; `decode_steps`
counts measured model forwards. Only counter reporting changed. **262 tests
pass** after reproducing all four failures. Full Mistral GPU validation covers
five policies, both prefill-logit settings, warmup and repeated trials, with
**390 observed forwards** following the exact token stream and positions.
The ordinary bulk path replays archived predictions, cache/state histories,
transfer bytes/calls and counter deltas. The owned GPU job finished. These are
schema/preservation checks; budget-matched quality/latency and full resource
equality remain open. The allocation audit above resolves the next diagnosis;
continue actual-resource comparisons after the stopped layerwise experiment. Full P0/P1/P2 remains
open; estimate **10–20 working days**, conditional on hardware/research gates.
[Verified profiler fix](results/hardware/baseline_profile_counters_v1/verified_report.json).

**Verified ordinary opt-in integration:** `--prefill-last-logits` is wired
through the LongBench evaluator, ordinary bulk/query evaluator, latency profiler
and fresh-memory caller. It requires explicit native `logits_to_keep` support;
the default remains disabled. The evaluator fingerprints the option and treats
legacy checkpoints without it as disabled. **252 tests pass**, including chunk,
EOS, token-limit, unsupported-model and profiler cases; all four CLI help checks
pass. Real ordinary-bulk Qasper runs with the flag off/on preserve the answer,
score and transfer/payload counts. Real fresh-memory CLI runs at both capacities
preserve the frozen prototype predictions/state/counters/transfers. The guarded
research memory results are **64.36% / 54.72%** reductions of active decode
scope; whole-request prefill remains **18.6278 GB**. Both monitored 3-ABBA timing
runs passed median/p95/TTFT/peak regression guards on a quiet GPU. All **24
actual API generation pairs** preserve 16 fields and replay **48 frozen rows**;
all owned jobs finished. The option is **adopted as an ordinary opt-in**, with
legacy-disabled checkpoint resume and enabled-mismatch rejection verified.
Shared/usual active envelopes remain **541.08 / 806.50 MiB**, and reserved scope
**5089.49 / 5345.49 MiB**; complete 512 MiB equality is not achieved. Full
P0/P1/P2 remains open. The fixed-teacher baseline counter error was
subsequently resolved in the current proof above; continue actual-resource
matched quality/latency work. Estimate **10–20 working days**,
conditional on hardware access and research gates.
[Verified ordinary integration](results/hardware/prefill_last_logits_integration_v1/verified_final_status.json).

- [x] GPU allocation diagnosis: final MLP multiplication and retained full
  prompt KV identified; frozen predictions/state/transfers/peaks and cleanup verified.
- [x] Preregister/test layerwise full-attention KV placement: whole-prefill primary
  passes10.30%; GPU-only decode guard fails2.31%. Freeze as research tradeoff;
  second geometry/generation/timing/adoption stopped.
- [x] Fresh faithful SnapKV/current ordinary actual-envelope audit:50 frozen
  prediction replays each,3200 observed layer widths, phase/source/cleanup proof.
  Both exceed512MiB and differ on49/50 width steps; matched P0 remains open.
- [x] Reserved SnapKV research baseline:512MiB genuine KV backing, exact
  CPU/GPU populated-bit/logit preservation, overflow/root/hook cleanup verified;
  complete522.61MiB active and3329.49MiB reserved scopes remain above target.
- [x] Packed real cold capacity: same256MiB pinned range/74.8% more occupiedKV;
 12development pairs fail direction gate−0.329; wider tests/adoption stopped.
- [ ] Establish actual resource matched quality/latency evidence for P0.

- [x] Native last-prompt-logit ordinary opt-in: both resource primaries,
  24 API generation pairs, actual bulk CLI pair, quiet 3-ABBA guards,
  checkpoint compatibility and 252 tests verified.
- [x] Fixed-teacher baseline profiler: all five policies work with exported
  counter schemas, observed decode counts and exact shared token streams;
  ordinary bulk replay and 262 tests verified.

[Verified prototype timing](results/hardware/prefill_last_logits_v1/verified_timing.json).

**Frozen last-prompt-logit pilot:** a preregistered research wrapper uses
Mistral's native `logits_to_keep=1` for prefill only. The frozen pilot left ordinary source/defaults
unchanged. Seven tiny-Mistral branch cases preserve outputs/cache/counters.
Two fresh GPU processes at 991/1090 capacities preserve all 50 predictions,
state, counters and transfers, replaying frozen ordinary controls. The active
decode envelope falls **1518.29 to 541.08 MiB (64.36%)**, passing the 10%
primary. Reserved scope falls **6089.49 to 5089.49 MiB**. Whole-request prefill
peak stays **18.6278 GB**, and complete 512 MiB equality is still unproved.
Usual capacities 1024/2048 also pass: **1781.31 to 806.50 MiB (54.72%)**;
reserved scope is **5345.49 MiB**. All **24 generation pairs** across six tasks
and both geometries preserve 16 fields and replay **48 frozen ordinary rows**.
All six untimed GPU jobs finished. Both preregistered monitored 3-ABBA TTFT/decode
checks subsequently passed on a quiet GPU; ordinary integration is verified
above. Full P0/P1/P2 and complete resource equality remain open.
[Preservation proof](results/hardware/prefill_last_logits_v1/verified_preservation.json).
[Verified shared resources](results/hardware/prefill_last_logits_v1/verified_resources_shared.json).

**Verified caller lifetime fix:** the ordinary generator releases the unused
prefill output after selecting its first token; the fresh teacher-stream caller
releases it before decode. A tiny Mistral regression reproduced the bug in
normal and chunked prefill; **243 tests now pass**. Two fresh GPU processes
preserve all 50 predictions, state, counters and transfers. Weak storage handles
confirm the **1000 MiB** prefill logits expire before the first decode.
The unchanged phase peak improves only **1536.64 to 1518.29 MiB (1.19%)**,
failing the 10% primary: its scope includes the handoff before output release.
Reserved scope remains **6089.49 MiB** and prefill peak **18.6278 GB**.
Keep the lifetime fix; stop its expansion as a phase-memory optimization.
All **24 archived/current generation pairs** across six tasks and both
geometries match in all 16 integrity fields and replay **48 frozen ordinary
bulk rows**. All four owned GPU jobs finished. Ownership prototypes remain
research-only. Local Mistral source supports computing only the last prompt
logits; that option is not implemented or measured. Preregister its independent
preservation/resource gates next. Full P0/P1/P2 completion remains open.
[Combined preservation proof](results/hardware/prefill_output_release_v1/verified_report.json).
[Verified caller resources](results/hardware/prefill_output_release_v1/verified_resources.json).

**Previous frozen caller portability decision:** the frozen ownership prototype
preserves all 50 fixed-stream predictions, final state, counters and transfer
bytes/calls in the fresh generation caller. Its active decode envelope falls
**1536.64 to 1410.90 MiB (8.18%)**, below the 10% primary; both exceed 512 MiB.
Reserved scope stays 6089.49 MiB and prefill peak stays 18.6278 GB. Both arms
retain 1000 MiB of prefill logits through the first decode. Stop usual-caller,
timing and ordinary integration expansion for this variant. The earlier
20.7–21.2% legacy-caller result remains valid within its narrower scope.
Next investigate unused prefill-output ownership in the generation caller.
Both owned audit jobs finished; frozen helpers and ordinary defaults unchanged.
[Verified fresh-caller decision](results/hardware/fresh_prefill_owner_v1/verified_report.json).

**Latest prefill ownership pilot:** all eight paired contexts preserve outputs,
state and transfers. Sharing the initial hot payload lowers the measured active
decode envelope **20.7–21.2%** at 991/1090 capacities and **14.0–14.1%** at
1024/2048. Shared active storage is about **411 MiB**; usual active storage is
**672–673 MiB**. Reserved scopes exceed 512 MiB in both geometries, and prefill
peaks are unchanged. **236 tests pass.** Mandatory timing checks remain; this is a serial research prototype with no ordinary/default change.
GPU 0 has an unrelated workload, so timing awaits a quiet measurement window.
[Verified pilots](results/hardware/prefill_owner_share_v1/verified_pilot.json).

**Generation preservation:** all 24 pairs across six tasks and both geometries
match the ordinary bulk path and replay all 48 frozen rows. All 1,213 candidate
forwards restore ownership and request hooks are removed. Reused development
data only; timing and complete-resource gates remain open. Scoped GPU lifetime
is verified below.
[Generation proof](results/hardware/prefill_owner_share_v1/verified_generation.json).

**Boundary/recycling preservation:** all 40 saturation and append cases and
six serialized interleaved/recycled requests pass. All 102 boundary frames
match; all 18 request rows replay frozen controls. Every scoped lease restores
its active request and hooks are removed. All owned preservation jobs finished.
Quiet-GPU timing remains open; scoped GPU lifetime is verified below. No
ordinary adoption or complete 512 MiB claim.
[Batch proof](results/hardware/prefill_owner_share_v1/verified_report.json).

**Scoped GPU lifetime gate passes:** 14 checks preserve every BF16/F32 payload
bit, untouched tails, released owners under allocation pressure, and GPU hook
restoration after normal and partial toy DynamicCache forwards. Two consumers
run serially with explicit caller events and `record_stream`. This does not
establish automatic stream handoff, concurrent serving or full Mistral exception
compute. Frozen prototype sources stay unchanged. Clean timing remains pending.
[Lifetime proof](results/hardware/prefill_owner_share_v1/verified_lifetime.json).

**Previous ownership decision:** forward-owner release preserves all four
shared-budget pairs and 1,536 layer releases/restorations. Active decode
envelopes fall **3.29–3.86%**, below the 10% gate, and remain **518–521 MiB**.
The 4K diagnostic locates the peak during first pre-forward validation;
release then removes about128 MiB of live GPU allocation. Reject advancement.
That diagnosis supported the prefill ownership pilot reported above.
**Suite at that milestone:227 tests pass.** [Verified decision](results/hardware/forward_owner_release_v1/verified_report.json).

**Previous memory decision:** the isolated raw-input-reuse prototype preserves
all four shared-budget context pairs, but decode active envelopes stay about
**539 MiB**; every 10% memory primary fails. A 4K phase audit confirms
384 bindings/compactions and about128 MiB released after forward. The peak is
already reached during forward (step2), so post-forward reuse is insufficient.
Stop before usual-budget/timing/generation/CLI expansion. **Suite at that milestone:
217 tests passed.** [Verified rejection](results/hardware/raw_input_reuse_screen_v1/verified_report.json).

**Earlier verified diagnosis:** 3,072 layer checks across 4K/8K/16K/31.5K
and both capacities confirm the incoming raw KV already contains the exact hot
prefix and has room for actual promotions plus new tokens. All 16 rows replay
frozen outputs/state/counters/transfers; **206 tests passed at that milestone**. This supports an
isolated buffer-reuse prototype, not a memory gain or safe in-place mutation
claim. [Prefix/ownership proof](results/hardware/decode_prefix_audit_v1/verified_report.json).

**Earlier screen decision:** charged per-token CUDA cache release is rejected.
It releases **104 MiB at the final boundary**, but lowers decode reserved
**peak by 0%**. The observed reserved envelope stays **519.74 MiB**, above
512 MiB. State, predictions, transfers and counters match; both host roots
release; **206 tests passed at that milestone**. Stop before timing/generation/CLI advancement.
[Verified decision](results/hardware/decode_reclaim_screen_v1/verified_report.json).

**Earlier resource audit:** six GPU 0 processes are finished and independently
verified. Bulk/control outputs, cache state and fresh-process GPU peaks match.
Current ordinary paths exceed the 512 MiB target: allocated decode envelopes
are about **539 / 804 MiB** in the legacy caller and **1537 / 1800 MiB** in the
fresh caller. Both reserved scopes exceed 6 GiB. The fresh caller retains
1000 MiB of prefill logits through its first decode; model-only allocator slack
is just 3.49 MiB. These callers differ, so their difference is not a causal
saving. No new memory/default claim; [resource proof](results/hardware/bulk_resource_audit_v1/verified_report.json).

**Latest verified implementation:** `--cpu-bulk-promotion` is available in
`experiments/evaluate_query_selection.py`, requiring `--cpu-reuse-visible` and
its existing CPU/direct/batched/fused/bounded/actual-query/LRU prerequisites.
Defaults stay unchanged. At 16K, three ABBA blocks per geometry improve the
same CPU backend: **65.97 → 62.52 ms/token (5.23%)** at 1024/2048 capacities,
and **63.31 → 59.17 ms/token (6.54%)** at 991/1090. All blocks improve; p95,
TTFT and measured GPU peak guards pass. Prefill peak stays **18.6278 GB**.

Preservation passes: 320 captured layer inputs, 50 boundary/fallback cases,
four model context lengths, 15 native bit/stream checks, 40 saturation cases,
24 generation pairs/six tasks, and six serialized recycled requests with
51 boundary frames per mode. All 48 generation rows reproduce earlier frozen
outputs/state/counters/transfers. Ordinary CLI samples match at both geometries;
eight invalid flag combinations reject before data/model access. Cached native
binaries load without Ninja. **Full suite: 206 tests pass.** All owned jobs
finished. These are CPU-control gains; quality, actual-resource equality,
faithful SnapKV speed, physical STT/controller and broader paper gates remain open.
[Stage](results/hardware/promotion_planning_screen_20261002/verified_stage.json),
[model parity](results/hardware/promotion_planning_screen_20261002/verified_model_parity.json),
[usual timing](results/hardware/promotion_planning_screen_20261002/verified_model_timing.json),
[shared timing](results/hardware/promotion_planning_screen_20261002/verified_shared_timing.json),
[integrity/replay](results/hardware/promotion_planning_screen_20261002/verified_integrity.json),
[ordinary CLI](results/hardware/promotion_planning_screen_20261002/cli/verified_report.json).

**Previous profile and stage screen (2 October 2026):** the ordinary native-reuse
policy profile identifies promotion/projection/victim preparation as a major
execution path. Its overlapping call times are diagnostic, not speed gains.
Standard-Torch metadata batching passes **18 exact numerical cases** and three
ABBA blocks, reducing the charged stage **4.52 → 3.05 ms**. The **1.47 ms saving**
misses the preregistered 4 ms threshold: stop before model integration.
The unused integration prototype is removed; measured helpers and frozen
source/raw evidence are preserved. No evaluator/default change. Full suite:
**199 tests pass**. Both owned GPU jobs finished; GPU 0 is idle.
[Profile proof](results/hardware/native_reuse_profile_20261002/verified_profile.json),
[stage decision](results/hardware/projection_batch_screen_20261002/verified_stage.json).

**Previous allocator-layout result (2 October 2026):** public snapshots
confirm fragmented CUDA segments. Four fresh pilots preserve state/outputs.
Shared-budget reservation improves **1.87%**, below the 10% primary; this arm
stops. Twelve fresh usual-budget timing processes verify **10.09% less decode
reservation**, but **median decode +2.78%** fails the 2% guard. p95 +0.78%,
TTFT +1.66% and unchanged prefill peak pass. Record this as a memory/latency
trade-off; stop advancement, with no ordinary CLI/default or global environment
change. Both full-resource ceilings remain failed/incomplete. Full suite:
**199 tests pass**. All 16 owned model jobs finished.
[Pilot/snapshots](results/hardware/allocator_layout_screen_20261002/verified_pilot.json),
[timing](results/hardware/allocator_layout_screen_20261002/verified_timing.json),
[decision](results/hardware/allocator_layout_screen_20261002/verified_decision.json).

**Latest caller-only research screen:** releasing unused prefill allocations
lowers **decode GPU reserved peak 20.60→16.57 GB (about 19.6%)**, against the
same exact-root policy. Three 50-step ABBA blocks pass both geometries;
TTFT rises 1.76% / 0.95%, median decode changes −1.62% / −1.85%.
Whole-request active peak stays **18.6278 GB**; no new speed/model-wide
memory claim. All 24 generation pairs, six recycled requests and 199 tests
pass; all 54 roots release. All 48 generation rows reproduce the earlier
frozen exact-root outputs/state. Candidate stays research-only because the
complete 512 MiB resource gate remains failed/incomplete.
[ABBA proof](results/hardware/prefill_reclaim_screen_20261002/verified_abba.json),
[integrity](results/hardware/prefill_reclaim_screen_20261002/verified_integrity.json),
[frozen replay](results/hardware/prefill_reclaim_screen_20261002/verified_frozen_replay.json).

**Latest storage candidate (2 October 2026):** exact-size pinned host roots
remove allocator excess: **256 → 144.25 MiB** at 991/1090 capacities and
**512 → 264 MiB** at 1024/2048. Four fresh processes preserve state, outputs
and counters; sampled peak RSS falls about **117 / 255 MB**. Whole-request
GPU peak remains **18.6278 GB**. This candidate remains research-only.

Nineteen allocation/lifetime checks, 768 multi-pool reads, 40 boundary cases
and four-context model parity pass. Suite at that milestone: **199 passing tests**.
Three ABBA blocks per geometry pass regression guards: median decode changes
**−1.44% / +0.12%**, TTFT changes **+4.59% / +7.77%**. These timings support
preservation, not a new speed claim. All 24 free-generation pairs/six tasks
and three recycled requests per geometry pass; 51 frames per mode match and
all 30 exact roots release. The full resource gate remains failed/incomplete:
active Torch decode envelopes are 427.40 / 556.25 MiB, and reserved scopes
exceed 512 MiB in both geometries. Candidate stays research-only; no CLI adoption.
Evidence: [exact-root memory](results/hardware/exact_pinned_pool_20261002/verified_memory.json),
[timing](results/hardware/exact_pinned_pool_20261002/verified_timing.json),
[generation/recycling](results/hardware/exact_pinned_pool_20261002/verified_integrity.json),
[whole-decode resources](results/hardware/exact_pinned_pool_20261002/verified_resources.json).

Latest verified speed change: the opt-in CPU coordinator plus fused hot-state
compaction reduces fixed-stream 16K decode from **75.47 → 66.44 ms/token
(11.96% faster)** at 1024-hot/2048-cold budgets. All three ABBA blocks improve;
p95 falls 85.02 → 72.17 ms, TTFT rises 0.46%, and prefill GPU peak is unchanged.
This compares the same CPU backend, actual-query policy, teacher stream and
history/synchronization instrumentation; it is not a new SnapKV comparison.

Exact state/IDs/counters/transfers pass at four context lengths, 32 synthetic
cases (fused and eager cold-overflow paths), 12 free-generation pairs across
six tasks, 15 native bit/stream-lifetime checks and three interleaved/recycled
requests with 51 boundary frames per mode. Ordinary CLI smoke and five
preload-error checks passed with **187 tests at that milestone**. Add
`--cpu-fused-compaction` to the actual-query direct CPU flags below. A verified
native binary loads without Ninja/recompilation in the measured Torch/Python
ABI. Combining with `--cpu-bounded-pool` is now verified at 1024/2048 and
991/1090 capacities: 40 native cases, four model lengths, three ABBA blocks,
12 free-generation pairs per geometry and 51 interleaved boundary frames per
mode match; all timing regression guards pass. Requested pinned roots fall
31.3% and 44.6%, respectively. Both flags can now be used together; defaults
stay unchanged. This does not establish an allocator/RSS or new speed gain.

The new shared-ceiling resource diagnostic finds **396 MiB at boundaries but
528.25 MiB of known live KV during fused compaction**, above the 512 MiB
ceiling. This already excludes metadata and retained allocator blocks, so
the transient-memory gate fails. The completed frozen SnapKV rerun is
**60.56 versus 25.83 ms/token (2.34× slower)**. Median, p95 and all-block
speed gates fail; TTFT +8.45% passes. This is a diagnostic comparison, not
proof of equal actual resources.

The whole-decode audit is complete: fused/bounded without reuse reaches 528.26 MiB
of known policy KV and SnapKV reaches about 520 MiB of classified KV, so
boundary snapshots alone bound neither transient peak. An opt-in
native raw-buffer reuse candidate lowers the same observed policy KV peak
**528.26 → 406.25 MiB (23.1%)**; post-decode boundary storage is 272.25 MiB.
This excludes forward temporaries and allocator blocks. The new public CUDA
allocation-range audit directly measures **256 MiB** behind the 144.25 MiB
shared root request, and **512 MiB** behind the 264 MiB usual root request.
Adding the shared root's extra 111.75 MiB to the SAME known-KV trace raises its
lower bound to **518.00 MiB**: the 512 MiB gate still fails. This still excludes
other cached blocks, OS rounding and GPU workspaces. Whole-request GPU peak
stays **18.6278 GB**, so no 10% model-wide memory gain is demonstrated.

Both candidate geometries pass four-context state/IDs/counters/transfers parity
and three ABBA regression blocks. Median decode is **1.34%/1.76% slower**,
within the frozen 2% tolerance; charged stage savings are only 0.26–0.34 ms,
below the 4 ms speed gate. Fifteen new native bit/stream checks pass, including
source and destination release under allocation pressure. The candidate
advances for storage only. All mandatory preservation gates now pass:
40 saturation/boundary cases, 24 free-generation pairs/six tasks across the
two geometries, and three recycled requests with all 51 frames per mode.
Ordinary CLI checks match at both geometries; seven invalid flag combinations
reject before model loading. Add `--cpu-reuse-visible` to the fused/bounded,
actual-query/direct/batched/LRU flags. The cached binary needs no Ninja.
Five cached-loader identity tests pass; full suite: **187 tests**.

Completed: fresh resource envelope audit of the ordinary bulk/native-reuse
implementation before selecting the next memory mechanism. Retain the
512 MiB ceiling and account for forward temporaries and allocator retention;
known policy KV alone is insufficient. Keep inspected quality holdouts excluded
from tuning. Full-resource equality, retention quality, faithful SnapKV speed,
physical STT/controller, second-model, throughput and paper gates remain open.
Overall estimate: **10–20 working days**, conditional on hardware access and
research outcomes.

Evidence: [fused stage](results/hardware/fused_hot_compaction_screen_20261002/verified_report_v2.json),
[model gates](results/hardware/fused_eviction_model_20261002/verified_report.json),
[native/generation/isolation](results/hardware/fused_eviction_stress_20261002/verified_report.json),
[CLI integration](results/hardware/fused_cli_integration_20261002/verified_report.json),
[combined systems](results/hardware/fused_bounded_pool_20261002/verified_systems.json),
[combined integrity](results/hardware/fused_bounded_pool_20261002/verified_stress.json),
[combined CLI](results/hardware/fused_bounded_pool_20261002/cli/verified_report.json),
[SnapKV timing and transient storage](results/baseline_fidelity/snap_fused_bounded_envelope_20261002/verified_report.json),
[reuse parity/resource proof](results/hardware/reuse_visible_model_20261002/verified_parity_resource.json),
[reuse model regression proof](results/hardware/reuse_visible_model_20261002/verified_timing.json),
[native/generation/saturation/isolation](results/hardware/reuse_visible_integrity_20261002/verified_report.json),
[reuse CLI and full suite](results/hardware/reuse_visible_integrity_20261002/cli/verified_report.json),
[public CUDA allocation-range audit](results/hardware/actual_reuse_memory_20261002/public/verified_report.json),
[prefill logits lifetime](results/hardware/actual_reuse_memory_20261002/logits/verified_report.json).

Earlier host-storage change: the opt-in bounded CPU pool reduces transient host
reservation while keeping logical cache capacities and token decisions intact.
At 991 hot / 1090 cold rows, owned pinned peak falls **44.6%**
(260.25 → 144.25 MiB); sampled peak RSS falls about 270 MB. GPU peak is
unchanged. Three ABBA blocks pass p95, TTFT and prefill guards; median decode
changes −0.70%, below the threshold for a speed claim.

Correctness passes at four model context lengths, 36 native boundary cases,
24 free-generation pairs across six tasks, 768 delayed pool/stream reads,
and three interleaved/recycled requests. CLI smoke and invalid-flag checks
pass; full suite: **154 tests**. Add `--cpu-bounded-pool` to the existing direct
CPU flags. Usual 1024/2048 geometry matches at four context lengths and in fresh
50-step processes. Requested root bytes fall 31.3%, but sampled peak RSS
changes by only 1.2 MB; both roots infer the same 512 MiB allocator block.
Three ABBA timing blocks and full 2112-slot native saturation pass their
regression/integrity guards. Median 75.22 → 75.17 ms is essentially unchanged.
No substantial host-RSS or GPU-peak gain is demonstrated at these budgets.
At these usual budgets, 12 free-generation pairs across six tasks also match
exactly (one greedy seed/native caps/EOS). Three interleaved/recycled requests
match isolated GPU and old-CPU controls at all 51 boundary frames per mode;
request roots reset cleanly. This closes these geometry-specific integrity gaps.
Serialized dispatch does not establish serving throughput or thread safety.

The research result remains negative against faithful SnapKV at the frozen
shared-ceiling configuration: the earlier direct path was **2.77× slower**;
the latest fused/bounded path is **2.34× slower**. Independent quality remains
**−2.57 points**
(CI [−5.29,−0.51]), 33 regressions and 10 losses over 10 points. These results
apply to 991/1090 logical capacities. The storage change preserves decisions;
it does not repair retention quality. Physical memory equality is unproven:
live tensor storage excludes allocator rounding, cached blocks and metadata.
The coordinator alone failed its 5% speed gate (3.09% gain). It is now used
only as part of the verified fused path above; default policies remain unchanged.

Latest diagnosis: cold overflow does not cause the fixed-stream decode gap;
it never occurs in 50 steps. Many small CUDA launches, metadata operations
and visible-cache assembly remain. Shared hot-index compaction saved only
2.13 ms, below the 4 ms screen; it was rejected for model integration.
The pooled-head maximum retention candidate also fails its frozen development
gate: **−0.60 points**, CI [−3.09,1.89], with a **17.27-point NarrativeQA loss**
on 12 reused documents across six tasks. Both methods answer the six synthetic
probes correctly; span retention changes only 154→156 layer/probe pairs. Three
ABBA prefill blocks at 16K/31.5K pass time and peak guards, but quality fails.
Candidate rejected; no untouched quality documents used and no default change.
The fused update/materialization candidate has now cleared its charged stage,
model speed and mandatory integrity gates as described above. Continue with
live-storage and visible-materialization diagnosis. The latest audit rules out
weight-only packing as a sufficient resource fix; investigate releasable
allocator blocks and persistent decode storage before proposing a mechanism.
Do not tune inspected holdouts.
The full project remains incomplete; estimated remaining effort is **10–20
working days**, contingent on hardware access and research gates.

Evidence: [bounded pool primary](results/hardware/bounded_pinned_pool_20261002/verified_report.json),
[stress and CLI](results/hardware/bounded_pinned_pool_20261002/verified_stress_report.json),
[usual-budget limits](results/hardware/bounded_pool_default_geometry_20261002/verified_report.json),
[usual-budget integrity](results/hardware/bounded_pool_default_stress_20261002/verified_report.json),
[CUDA diagnosis](results/hardware/full_decode_trace_20261002/verified_trace_analysis.json),
[rejected compaction screen](results/hardware/shared_hot_index_screen_20261002/verified_report.json),
[retention pilot](results/baseline_fidelity/head_max_retention_dev_20261002/verified_report.json),
[rejected broader retention](results/baseline_fidelity/head_max_broader_dev_20261002/verified_report.json),
[SnapKV systems](results/baseline_fidelity/snap_tier_owned_envelope_20261002/verified_timing.json),
[independent quality](results/baseline_fidelity/snap_tier_envelope_fresh_quality_20261002/verified_report.json).


The direct CPU initializer is now integrated and verified. Use
`--strategy lru --real-query-prefill --cpu-batched-scores --cpu-direct-shared`
with `experiments/evaluate_query_selection.py` and its usual data/task/output
arguments. GPU and existing CPU defaults remain unchanged; the CPU path still
requires the locally verified Torch2.5 CUDA transfer binaries.

1. **Memory:** direct initialization halves peak owned pinned buffers
   (805.6 → 402.8 MB), reduces allocations from 33 to one, and lowers sampled
   process RSS by about 544 MB. Peak GPU allocation is unchanged. Allocator
   rounding capacity is inferred from source/trace, not direct host telemetry.
2. **Integrity:** 4K/8K/16K model parity, 32 synthetic cases, 768 delayed
   pool/stream checks, and three interleaved model requests pass. Resetting
   one request and creating another preserves the still-live request. These
   are serialized model checks, not threaded serving or throughput proof.
3. **Timing:** versus the existing CPU backend, TTFT rises 2.13% (within
   the 10% guard); median decode changes −0.18% and p95 +0.10% (within 2%).
   This initializer is a host-memory improvement, not a new decode-speed claim.
4. **Quality:** the earlier 100-document/five-task study improved +5.87 points
   (CI [3.16, 8.83]), but 17 documents regressed. Adapted NarrativeQA remains
   inconclusive: +0.36 points (CI [−7.80, 9.06]), six regressions, four losses
   over 10 points; four of 20 original sources failed without replacement.
   Keep the combined path opt-in.
5. **Still open:** the faithful SnapKV decode gap and actual-resource/width
   comparison, second model, throughput/saturation, physical STT/controller
   evidence and final independent paper audit. Overall research work is not
   complete; estimated remaining effort is 10–20 working days, contingent on
   hardware access and research gates.

Evidence: [direct initializer](results/hardware/direct_shared_init_20261001/verified_report.json),
[CLI and interleaved integration](results/hardware/direct_integration_20261001/verified_report.json),
[fresh-process memory](results/hardware/fresh_process_memory_20261001/verified_report.json),
[NarrativeQA](results/baseline_fidelity/narrativeqa_conditional_quality_20261001/verified_report.json).
50-step CUDA trace completed: no cold overflow (1616/2048 rows).
Shared hot-index screen saved 2.13 ms, below the 4 ms threshold; rejected.
GPU 0: both four-context prefill-owner pilots finished and verified. Nine focused fixtures and 236 tests pass. Previous six model preservation jobs finished and verified:24generation pairs,40boundary cases,6recycled requests/102frames. Fourteen scoped GPU bit/owner/exception checks finished and independently verified. Fresh GPU0 preflight finished: no sustained quiet interval across20samples. Clean timing remains pending. Both fresh-caller audit jobs finished and verified:8.18%active-envelope saving fails10%primary. Stop fresh-caller expansion; investigate retained prefill-output lifetime next. Unrelated workloads remain untouched; no ordinary/default adoption.
Shared arm fails its 10% memory pilot. Usual memory primary passes10.09%, but
median decode+2.78% fails2% guard. Advancement stops; research diagnostic only.
Suite at that allocator milestone: 199 passes. Current suite: 236. Quality/resource/physical-tier/P0/P1/P2 remain open.
Three ABBA blocks and generation/recycling pass at both geometries. Decode
reservation falls about 19.6%, whole-request active peak unchanged. The
complete 512 MiB resource target remains failed/incomplete; research-only.
Exact-root allocation, four-context parity, fresh memory, ABBA, generation and
recycling proof pass at both geometries; 199 tests passed at that milestone.
Full resource diagnostic is complete: active envelope 427.40 / 556.25 MiB;
usual active and both reserved scopes fail 512 MiB. Candidate stays
research-only; actual-resource equality and P0/P1/P2 remain open.
Earlier public CUDA audit verified.
Native reuse CLI/integrity stays verified; 187 tests passed at that milestone.
Root requests144.25/264 MiB own CUDA host ranges 256/512 MiB. Shared corrected
known-KV lower bound 518.00 MiB exceeds512; whole GPU peak 18.6278 GB unchanged.
Twelve fresh memory processes (eight private-reader plus four public-API),
six public/private size/alias checks and two weak logits runs are archived.
Active host probe uses public CUDA API; private reader source removed,
compiled host binary needed only for optional historical cross-check.
Defaults unchanged; full research P0/P1/P2 remain open.

**Updated: 3 October 2026.** This is the short status page for the student
and supervisor. Check this page first; detailed experiment history stays in
the linked files. A checked box means the stated work was done, not that the
research contribution is complete.

**New to the project?** Read [the short first page](START_HERE.md), then
[the longer plain-language guide](SIMPLE_NOTES.md) if needed. You only need
their first sections to follow this
checklist. **P0** means needed before we can make the paper's main claim;
**P1** means evidence that strengthens or tests that claim; **P2** means
finishing the paper and reproducibility package. An unchecked box means the
evidence is still missing, even if some code or pilot tests already exist.

**Document map:** Use `START_HERE.md` to learn the idea, this page to see
what is done, and `explanation.md` to inspect the code in detail. The long
roadmap and experiment log are evidence records; you do not need to read them
from start to finish. Older explanation and audit files are historical records.

## Current state in plain language

The project suite passes **306 tests**, including19 new visible-suffix tests.
Previously adopted bulk promotion and last-prefill-logit options remain optional;
ordinary source, CLIs and defaults are unchanged. New query work remains research.

Visible-buffer reuse preserves every current-query result checked and improves
the GPU-index query component **7.85%** (105.32→97.05ms). Actual fresh active
memory is545.17MiB, still above512MiB. KV H2D remains5.826GB/50steps plus75.33MB
of index metadata. The fresh ordinary-LRU comparison remains slower at
**1.686× median /1.559× p95**; stop standalone suffix-query expansion/adoption.
Both controlled comparisons replay frozen state/predictions/counters/native
bytes/calls and metadata. All four ownGPU jobs finished; roots/hooks/index
metadata release. Next preregister keys-only ranking reads and selected-value
reads with unchanged ranking math and separate correctness/resource/cost gates.

The prior diagnostic summary localizes composition overhead but has no complete
Chrome/native-kernel trace; its original failed reporting attempts remain intact.
Projection reuse alone, the fixed8-page heuristic and packed-cold capacity remain
stopped on their respective failed gates. Positive component improvements do not
prove a joint quality/latency/resource win.

The ordinary resource comparison remains unequal: active541.08MiB TieredKV /
528.74MiB faithful SnapKV; reserved5089.49/2881.49MiB; prefill18.6278/16.9583GB.
Attention widths differ on49/50 steps and both complete active envelopes exceed
512MiB. Reserved SnapKV also exceeds the complete target; the layerwise-prefill
candidate failed its decode-memory guard. Actual matching is unproved.

Older independent DirectCPU quality fails NI (−2.57points,CI[−5.29,−0.51]);
inspected query development data is not independent quality evidence. Physical
STT/controller/energy/endurance, second-model/robustness/serving evaluation and
supported final paper/reproducibility remain open. Remaining effort:
**10–20 working days**, conditional on hardware and research outcomes.

## Supervisor update (copyable)

306 tests pass; ordinary source/CLI/defaults are unchanged. Retaining visible
buffers and replacing only selected cold rows exactly preserves checked query
behavior and saves7.85%/8.27ms in controlled component timing. Fresh resource
use545.17MiB still exceeds512MiB, and coldKV/index traffic is unchanged. Fresh
ordinary-LRU guards fail1.686×median/1.559×p95 despite passingTTFT. Stop standalone
suffix-query expansion/adoption; keep the component as research evidence. All
four ownGPU jobs finished with hooks/index/owned roots released. Next preregister
keys-only ranking reads and selected-value reads, retaining existing query math.
Full512MiB, actual faithful-baseline matching, independentquality, physicalSTT and
remaining P0/P1/P2 remain open. Estimate10–20 working days, conditional on hardware/
research outcomes.

## Completed and verified

- [x] **Visible cold-suffix composition gates:**19CPU/306suite pass; exact
  GPU4K12 frozen-query replay and384 retained destinations/prefix ownership
  checks. Fresh16K50 active545.17MiB fits576pilot; bytes/calls/metadata replay
  and owned cleanup pass. Component3ABBA saves7.85%/8.27ms; freshordinaryLRU
  guard fails1.686×median/1.559×p95. Stop standalone expansion/adoption;
  four GPU jobs terminal. [Decision](results/hardware/query_visible_suffix_v1/verified_lru.json).

- [x] **Scoped query-phase diagnostic summary:**4CPU parity cases and287suite
  pass;8CPU ranges ×1600 observations,6duplicate GPU spans excluded. Fixed
  query preservation assertions and5quiet-process checks completed; ownhooks/
  index metadata released and all jobs terminal. Visible composition15.89ms
  instrumented CPU range is largest; no complete Chrome/native-kernel trace
  or speed/adoption claim. [Summary](results/hardware/query_phase_profile_v1/verified_summary.json).

- [x] **BatchedGPU native query read-index gates:**10CPU/283suite pass,
  3native dtypes/stream/pool/stale/cleanup cases, exact4K12/full16K50 native
  query replay. Active547.488MiB fits576pilot; metadata75.33MB separately
  charged. Component3ABBA saves24.61%median, but freshordinaryLRU guard fails
  1.810×median/1.679×p95. Stop standalone full-query expansion/adoption;
  unchangedordinaryCLI/defaults. [Decision](results/hardware/gpu_query_read_indices_v1/verified_lru.json).


- [x] **Native query-projection reuse experiment:**7focusedCPU/273suite pass,
  fullGPU4K12 nativeCPU predictions/logits/cache/counters/choices/transfers
  exactly replay. Projection calls768→384; ownhooks/pendingcontext clear.
  Quiet3ABBA visible-only cost82.129→81.559ms saves0.694%, below5%primary;
  p95/TTFT guards pass. Stop standalone expansion; research-only with no
  ordinary/full-scan adoption. [Decision](results/hardware/reused_query_projection_v1/verified_cost.json).


- [x] **Visible-only query-hook cost diagnosis:** quiet3ABBA16K50,300measured
  forwards/arm, exact ordinary predictions/cache/logical counters/transfer
  bytes, no extra cold reads and ownhooks removed. Existing unfused pattern
  adds40.60%median/35.79%p95 latency; TTFT+0.68%passes. Reduced traffic alone
  cannot rescue that pattern; no universal lowerbound/adoption claim.
  [Proof](results/hardware/query_hook_cost_v1/verified_report.json).


- [x] **Native-trajectory page-pruning screen:**1600 observations exactly
  replay full16K50 native-query state/predictions/counters. Eight centroid pages
  cover13.49%live rows but retain28.39%top32 and55.40%optimistic attention mass.
  Predeclared recall screens fail; stop this fixedheuristic's implementation/
  quality/timing expansion. Ownhooks/canonicalguard restore. [Proof](results/hardware/native_page_recall_v1/verified_report.json).


- [x] **Native-pool current-query research gate decision:** four tiny pairs,
  266tests and three GPU implementations preserve original query predictions,
  state and visibleKV. Fresh544.321MiB active fits576MiB pilot; allroots/hooks
  release. Development12pairs:+2.132points, but full-scan5.826GBH2D/50steps and
  quiet3ABBA median/p95 regress2.383×/2.215×. Stop widerquality/timing/adoption;
  ordinaryLRU remains control. [Proof](results/hardware/pool_query_selector_v1/verified_timing.json).


- [x] **Fresh-caller portability falsification:** two 16K/50-step processes preserve predictions/state/counters/transfer bytes and calls, and replay frozen ordinary output. All 50 owner leases restore 1600 layers; both roots release. Active envelope1536.64 to1410.90MiB saves8.18%, failing10%primary and512MiB. Both retain1000MiBprefilllogits through first decode. Stop usual/timing/ordinary expansion for this caller. [Proof](results/hardware/fresh_prefill_owner_v1/verified_report.json).

- [x] **Scoped GPU stream/lifetime proof:**10 BF16/F32 gather/owner-pressure cases and4 GPU normal/partial exception cases pass with explicit caller handoff and two serial consumers. All frozen source hashes match; no ordinary change. Clean timing and complete-resource equality remain open. [Proof](results/hardware/prefill_owner_share_v1/verified_lifetime.json).

- [x] **Prefill ownership preservation batch:**24generation pairs replay48frozen rows;40boundary cases and6recycled requests pass with102exactframes/18frozenrequestrows. All owned jobs finished;236-test suite passed. Timing/stream-lifetime/full-resource/ordinary-adoption gates remain open. [Proof](results/hardware/prefill_owner_share_v1/verified_report.json).

- [x] **Prefill ownership memory pilot:** eight pairs across both geometries pass the 10% active-envelope primary and exact frozen replay. Shared active storage is about 411 MiB; usual active and both reserved scopes exceed 512 MiB. Nine focused cases and 236 tests pass. Timing, generation/lifetime/recycling and ordinary integration remain open. [Proof](results/hardware/prefill_owner_share_v1/verified_pilot.json).

- [x] **Forward ownership falsification:**10focused cases prove partial
  exception restoration, old owner expiry, row-padded inputs and fallbacks.
  Four model context pairs match outputs/state/counters/transfers and replay
  all8frozen rows; all1,536layer releases/restorations complete. Active
  envelopes518–521MiB improve3.29–3.86%, below10%primary and still above512.
  The4K phase audit locates peak in first validation before release and
  measures127.81MiBlive release. Reject usual-budget/timing/generation/CLI
  advancement;227tests pass. No ordinary/default or new kernel.
  [Decision](results/hardware/forward_owner_release_v1/verified_report.json).

- [x] **Raw-input payload-reuse falsification:** standardTorch per-layer
  scratch gathers preserve K/V/metadata in11focused fixtures and four real
  context pairs. All8rows replay earlier frozen outputs/state/IDs/counters/
  transfer/root data. Measured active envelopes stay~539MiB, failing10%memory
  primary and512MiBtarget. The4K diagnostic confirms384bindings/compactions,
  ~128MiBpost-forward release and peak already reached in forward step2.
  Reject before usual-budget/timing/generation/CLI expansion.217tests pass;
  no ordinary/default or new native kernel. Full-resource equality stays open.
  [Proof](results/hardware/raw_input_reuse_screen_v1/verified_report.json).

- [x] **Hot-prefix ownership/capacity diagnosis:** two fresh GPU processes,
  four contexts per geometry and 3,072 layer checks verify bit-identical
  incoming hot prefixes, distinct old owners, extension capacity and unchanged
  new-token tails. All16rows reproduce frozen outputs/state/counters/transfers.
  Old hot owners expire after ordinary preparation when promotion occurs.
  Historical peak groups sum exactly: incoming raw128 MiB and concatenated
  KV~130 MiB. This is descriptive attribution, not causal memory saving.
  Isolated prototype allowed; alias-safe mutation, timing/peak/stress and
  broader gates remain unproved. Full suite206passes.
  [Proof](results/hardware/decode_prefix_audit_v1/verified_report.json).

- [x] **Ordinary bulk resource audit:** six finished GPU processes verify
  within-caller outputs/state/counters/payloads and equal fresh GPU peaks.
  Actual host ranges and public snapshots confirm ordinary observed scopes
  exceed 512 MiB. Model-only slack 3.49 MiB rules out weight-only packing
  as a sufficient fix. Full-resource equality stays open.
  [Evidence](results/hardware/bulk_resource_audit_v1/verified_report.json).

- [x] **Per-token cache-release falsification:** the shared research pilot
  preserves state/IDs/transfers but releases only final-boundary reservation
  (104 MiB); peak saving is 0%. Reject before full timing/generation/CLI.
  Both roots release; full suite 206 passes. Defaults unchanged.
  [Decision](results/hardware/decode_reclaim_screen_v1/verified_report.json).

- [x] **Bulk CPU promotion planning:** 5.23% / 6.54% same-backend gains at
  usual/shared capacities; all timing and integrity guards pass. Ordinary flag
  `--cpu-bulk-promotion` requires the native-reuse chain. Two-geometry CLI
  parity, eight invalid combinations and 206 tests pass. Defaults unchanged;
  broader quality/resource/hardware gates remain open.
  [Evidence](results/hardware/promotion_planning_screen_20261002/cli/verified_report.json).

- [x] **Public allocator snapshot/layout screen:** four fresh pilots and
  twelve usual-budget fresh timing processes verify exact state/IDs/counters/
  payloads. Snapshots distinguish trapped free bytes from unused segments.
  Shared pilot fails 10% primary; usual gains10.09% reservation but fails median
  latency guard (+2.78%). Stop progression, archive the trade-off and keep
  ordinary settings unchanged. Full suite 199 tests. No candidate generation/
  lifetime expansion after this failed timing guard; earlier native/source
  proofs remain historical, not evidence of this candidate’s serving safety.

- [x] **Retained-prefill CUDA cleanup research screen:** charged once inside
  TTFT, with the exact-root policy unchanged in both arms. Decode reserved
  peak falls about 19.6%; three-ABBA regression guards pass. All 24 generation
  pairs/six tasks, six recycled requests, 51 frames per mode and 199 tests
  pass; all 54 exact roots release. Frozen-control replay matches across
  all 16 preservation fields. Active/model-wide memory and full resource
  gates remain open; no ordinary CLI/default promotion.

- [x] **Exact pinned-root research candidate:** actual host ranges shrink
  43.65% / 48.44%; sampled peak RSS falls about 117 / 255 MB. Allocation,
  native/boundary, four-context, three-ABBA, 24 generation pairs and six
  recycled requests pass; all roots release. Full suite: 199 tests. One
  counter-hook wrapper error was fixed, tested and rerun; failed raw evidence
  remains archived. Whole-resource proof keeps the candidate research-only:
  427.40 / 556.25 MiB active envelopes; reserved scopes fail both budgets.
  This completes the candidate’s diagnosis, not the research memory gate.

- [x] **Pinned allocation-range and logits audit:** public CUDA API matches
  six archived private capacity/alias checks; four new fresh-process runs
  match the eight prior memory runs in state/IDs/counters and GPU peaks.
  Shared root256 MiB; usual root 512 MiB. Corrected shared KV lower bound
  518.00 MiB fails 512 MiB. Two weak-storage runs confirm 1000 MiB prefill logits
  remain live through the first decode forward in the fresh-memory harness.
  This completes diagnosis, not the broad memory/research gate.

- [x] **Native raw-buffer reuse, opt-in storage tradeoff:** known policy KV
  peak falls23.1%; whole-request GPU peak unchanged; median decode1.34–1.76%
  slower within the 2%regression limit. Four-context/both-geometry parity,
  native owner-release, saturation, six-task generation, recycling and CLI
  checks pass;187tests. Flag `--cpu-reuse-visible` requires fused/bounded CPU
  configuration. No actual-resource equality, speed or quality claim.

- [x] **Bounded/fused composition:** both budget configurations pass native,
  four-length parity, ABBA regression, six-task generation, request isolation
  and ordinary CLI gates. Both opt-in flags can be combined; defaults stay
  unchanged. Frozen SnapKV timing and transient-memory diagnostics are
  complete, and their superiority/512 MiB gates fail.
- [x] **Review fixes and corrected control:** latency includes cache-policy
  work; Mistral evaluation handles chat/EOS and datasets correctly; output
  provenance and figure/table inputs are checked. The 14-job corrected matrix
  finished in [`results/remediation_v3/`](results/remediation_v3/). Details:
  [`REMEDIATION_CHECKLIST.md`](REMEDIATION_CHECKLIST.md).
- [x] **Implementation audit:** [`explanation.md`](explanation.md) identifies
  actual GPU residency, token lifecycle, policies, transfers, limitations,
  and which STT-RAM properties are modeled.
- [x] **Stronger baseline check:** 135 source-matched requests against an
  upstream-selector SnapKV adapter were evaluated. Quality superiority or
  equivalence was **not** established; actual resident bytes and attended
  widths are not yet matched. Evidence:
  [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md),
  [`results/baseline_fidelity/`](results/baseline_fidelity/).
- [x] **Shadow-cache causal check:** paired inclusive/no-shadow evaluation
  found 88.1% fewer logical paid decode-demotion writes per step across 135
  requests; four fixed-stream ABBA pairs counted 2,690 versus 26,432 writes
  on the same 50 decode inputs (89.82% fewer). An opt-in CPU-memory replay
  also copied 11.02 versus 108.27 MB of decode K/V payload to CPU on that
  stream. This is **not** physical STT-RAM page traffic or device lifetime.
  Evidence:
  [`results/shadow_ablation_full/`](results/shadow_ablation_full/) and
  [`results/hardware/cpu_shadow_copy_16384_50_20260929.json`](results/hardware/cpu_shadow_copy_16384_50_20260929.json).
- [x] **Controlled candidates and rejection decisions:** bounded prefill cut
  peak allocation but has unresolved quality; actual-query placement showed
  factual regressions and prefill cost; a 50/50 selector mix and several
  local decode optimizations failed their gates. Rejected variants were not
  promoted to the default. Details: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
- [x] **Research process:** comparison gates, stress checks, experiment log,
  working paper, and a conditional submission schedule exist. The paper is
  a draft, not a finished submission.
- [x] **FlexiCache resource and transfer preflight:** audited the frozen
  release and compiled its unchanged CPU/GPU transfer kernels for L40S.
  Exact fp16/bf16 K/V, completed-page mapping, unstable-head exclusion,
  empty ranges and background-stream checks pass. The single-prompt
  analytical payload is 618.04 MB GPU plus 1,550.84 MB CPU, versus
  TieredKV's 402.65 MB logical ceiling; these are unequal resources.
  The default pinned pool is 180 GiB; allocation now passes in the native
  engine. One identical-token inference run now passes with recorded L40S
  runtime/launch adapters, including a fresh-process repeat. Evidence:
  [`resource preflight`](results/baseline_fidelity/flexicache_resource_preflight_v2_20261001.json)
  and [`transfer check`](results/baseline_fidelity/flexicache_transfer_check_v2_20261001.json).
- [x] **FlexiCache native build and import:** recovered the interrupted build
  from retained objects, installed its unchanged native targets, and passed
  the exact-source/version import gate on L40S. Model execution is a separate
  separate gate; the first adapted model request has now passed.
  Evidence: [`native import`](results/baseline_fidelity/flexicache_engine_import_20261001.json).
- [x] **FlexiCache adapted same-input feasibility:** two fresh L40S processes
  reproduce the frozen 1,440-token request, generated IDs/text, score 10/100,
  bf16 pools and submitted transfer payload. Scheduled-token allocation
  traces match; idle-poll counts differ. This is a documented runtime/launch
  port with unequal resources, not a quality or speed comparison. Evidence:
  [`pair validation`](results/baseline_fidelity/flexicache_matched_input_pair_report_20261001.json).

- [x] **Actual cache ownership trajectory:** measured three arms on identical
  16K input and 50 teacher-forced decode tokens. All 51 boundaries count
  distinct KV allocations once, including DynamicCache attention copies
  and the shared pinned CPU pool. GPU/CPU Tiered row geometry and predictions
  match exactly. The verified storage equations and widths reject resource
  equality: final SnapKV 512 MiB GPU, Tiered GPU 462.5 MiB, CPU backend
  260 MiB GPU + 384.125 MiB host reservation (202.508 MiB host occupied).
  This is accounting evidence, not a quality, latency, peak-process-memory,
  matched-resource or physical-STT gate. Evidence:
  [`resource trajectory`](results/baseline_fidelity/snap_tier_resource_trajectory_verified_20261001.json).

## Remaining work, in priority order

- [ ] **P0 — Demonstrate a credible physical tier.** Move cold K/V to a
  CPU-backed tier or another implementable memory interface; measure real
  transfer bytes, occupancy, synchronization, and whole-loop cost. Current
  GPU-resident default cannot support physical STT-RAM claims. An opt-in
  CPU-DRAM probe preserved policy/output and saved 219 MB final GPU allocation
  at 16K, but slowed decode and left prefill peak unchanged. A faster native
  CPU gather passed three 50-step paired checks, yet the CPU path remains
  slower than the GPU prototype; a batched/overlapped design and hardware
  traffic/energy evidence are still needed. Keep the
  agreed STT-RAM bandwidth parameters unchanged.
  A corrected pinned-slot storage screen passed exact K/V checks and three
  ABBA blocks: 6.759 versus 13.891 ms for storage operations, with 272.76 MB
  reserved versus 203.95 MB occupied host payload. This justified a real-cache
  trial, whose full-loop speed gate subsequently failed.
  Evidence: [`slot screen`](results/hardware/cpu_slot_transfer_screen_v2_20261001.json).
  The experimental slot cache passed synthetic lifecycle checks and exact
  4K/8K/16K five-step Mistral parity for cache state, selections, predictions,
  counters and transfer payload. Its real-cache pool reserves 402.78 MB
  for transient headroom. Three 16K/50-step ABBA blocks preserved all state,
  predictions and payload, but improved median decode only 0.35%
  (88.918 versus 89.228 ms), below the 5% gate. The candidate and its active
  hooks were removed; exact executed sources and results are archived.
  Evidence: [`full-loop rejection`](results/hardware/cpu_slots_abba_16k_50_20261001.json) and
  [`model parity`](results/hardware/cpu_slots_model_parity_8k_16k_20261001.json).
  Cross-layer native transfer batching now passes a charged storage screen:
  median 2.323 versus 6.707 ms, saving 4.384 ms; all three ABBA blocks improve
  and every read/final K/V matches exactly. This is enough to investigate
  model integration, but assumes all layer inputs and selections are available
  together. The model candidate below now tests actual scheduling and parity.
  Evidence: [`batching screen`](results/hardware/cpu_cross_layer_transfer_screen_20261001.json).
  An opt-in shared-pool candidate now passes four two-layer synthetic regimes
  (24 steps each), including exact state/payload, promotion floors, empty and
  repeated reads, compaction/reuse ordering and reset with queued writes.
  4K/8K/16K five-step Mistral parity now passes: exact cache state at every
  visibility boundary, selections, predictions, counters and transfer payload.
  The packed reference also matches earlier frozen runs from before the new
  hooks. The 16K/50-step three-block ABBA gate now passes: **80.017 versus
  90.685 ms/token (11.76% faster)**, p95 89.241 versus 109.218 ms.
  All twelve arms preserve state, selections, predictions, counters and useful
  transfer payload. Retained as an opt-in backend. Final host tensors reserve
  402.78 MB for 212.34 MB occupied; initial consolidation briefly owns up to
  805.57 MB of host tensors, excluding allocator retention. Median TTFT
  increases 3.98%; prefill GPU peak is unchanged and decode peak rises 3.09 MB.
  GPU-versus-CPU 4K/8K/16K five-step parity also passes. The fresh GPU control
  completes all twelve 16K/50-step arms: 79.941 ms/token batched CPU versus
  81.515 GPU, but the 1.93% pooled gain fails the 5% gate and two of three
  blocks favor GPU. CPU p95 is 0.65% worse, within its guard. Final GPU
  allocation falls by 229.91 MB (1.55%), below the 10% total-memory gate;
  prefill peak is unchanged. No promotion over the GPU prototype. Retain
  opt-in for the passed CPU-control gain; broader workload/concurrency and
  physical STT/controller evidence remain open.
  Evidence: [`paired GPU control`](results/hardware/cpu_batched_gpu_gate_report_20261001.json).
  Evidence: [`synthetic batching`](results/hardware/cpu_batched_synthetic_v2_20261001.json),
  [`model parity`](results/hardware/cpu_batched_model_parity_20261001.json),
  [`full-loop gate report`](results/hardware/cpu_batched_gate_report_20261001.json).
  Suite at that milestone: **111 passed**, one existing dependency warning.
- [ ] **P0 — Close the decode gap.** Remove structural per-layer policy and
  visible-cache overhead, then rerun same-source, fixed-output ABBA latency,
  p95, memory and quality checks against faithful SnapKV. A small isolated
  kernel/mask speedup is insufficient; the earlier measured gap is 3.21×.
  A new policy-only profile of the batched backend puts eviction/demotion,
  repeated tensor construction and score updates above grouped transfers.
  These diagnostic times include implicit GPU waits and are not additive
  savings. Next candidate must pass a charged screen before a full-loop test.
  Evidence: [`policy profile`](results/hardware/cpu_batched_policy_profile_20261001.prof.txt).
  Cross-layer attention-score bookkeeping passes a charged exact screen:
  1.596 versus 6.386 ms (4.790 ms saved), including fp32, zero-cold,
  remapped-position and heterogeneous fallback checks. An opt-in policy
  passes every boundary-state/selection/output/byte check at 4K/8K/16K.
  The three-block 16K/50-step whole-loop gate now passes: **75.560 versus
  80.956 ms/token (6.66% faster)**, p95 84.856 versus 89.494 ms. All twelve
  arms preserve state, selections, predictions, counters, transfer payload
  and native dispatch counts. Prefill/decode GPU peaks and final GPU/host
  allocations match; median TTFT rises 0.85%. Retain opt-in; defaults unchanged.
  Fresh GPU control also passes: **75.646 versus 80.475 ms/token (6.00%
  faster)**; all blocks improve, p95 falls from 88.883 to 84.982 ms and
  exact state/output/payload identities match frozen controls. Final GPU
  allocation falls 229.91 MB (1.55%); prefill peak is unchanged and median
  TTFT rises 4.23%. Unequal host reservation remains explicit; this is not
  a matched-total-resource or SnapKV comparison. Mandatory remaining stress
  and baseline checks keep the candidate opt-in and the broad P0 item open.
  Evidence: [`score screen`](results/baseline_fidelity/batched_score_screen_20261001.json)
  and [`score model parity`](results/hardware/batched_scores_model_parity_20261001.json),
  [`score full-loop gate`](results/hardware/batched_scores_gate_report_20261001.json),
  [`score GPU-control gate`](results/hardware/batched_scores_gpu_gate_report_20261001.json).
  Free-generation stress now passes 24 pairs: 12 distinct reused frozen
  documents across QA/summarization and seeds 0/1, with actual task generation
  limits, EOS/cap stops, 11–449 generated tokens and prompts up to 21,202
  tokens. Every ID, score, full final state and visibility trajectory matches.
  Paired task-balanced quality difference/interval is 0 / [0,0]; this proves
  preservation on these cases, not better or unseen-corpus quality. Largest
  final cold occupancy is 267.44 MB, near its 268.44 MB logical limit; a full
  saturation/burst proof was subsequently completed in explicit boundary
  cases below. Another model, process peaks and concurrency remain open.
  Evidence: [`verified generation stress`](results/hardware/batched_scores_generation_stress_20261001/verified_report.json).
- [ ] **P0 — Establish a defensible claim at matched resources.** Compare
  faithful baselines at ex-ante equal actual resident KV bytes and comparable
  attended width; report quality, prefill, decode, peak/steady memory,
  throughput, and writes together. Do not state an accuracy or SOTA win from
  the current overlapping confidence interval.
  The corrected 2 October audit verifies a shared 512 MiB boundary-owned KV
  ceiling (including visible copies and pinned reservation) and 1024-row
  single-token attention ceiling across 153 boundaries. Actual use differs:
  Snap 487–512 MiB; GPU Tiered 446.875–456.875 MiB; direct CPU Tiered 512 MiB.
  At these frozen ceilings, three ABBA blocks show CPU Tiered 71.20 versus
  Snap 25.74 ms/token (2.77× slower), p95 78.61 versus 25.96 ms. TTFT +7.96%
  passes; prefill peak rises 16.96 → 18.63 GB. Independent five-task quality
  on 100 unused documents is −2.57 points versus SnapKV (CI [−5.29,−0.51]);
  NI and superiority fail, with 33 regressions and 10 losses over 10 points. This is a shared-ceiling diagnostic,
  not equal actual bytes/width, a joint win, or completion of this checkbox.
  That historical DirectCPU tensor-storage ceiling excluded allocator rounding:
  its260.25MiB root occupied an inferred512MiB block. The current bounded/bulk
  fresh audit instead measures a256MiB owned pinned block and285.08MiB GPU
  decode allocation excluding model storage:541.08MiB total, versus faithful
  SnapKV528.74MiB. Widths differ on49/50 steps; complete equality remains
  unproved. Unowned host/driver scopes and physicalSTT remain unmeasured.
  Current evidence: [actual-envelope audit](results/hardware/snap_actual_envelope_v1/verified_report.json).
  Evidence: [`owned-KV audit`](results/baseline_fidelity/snap_tier_owned_envelope_20261002/verified_report.json),
  [`paired systems result`](results/baseline_fidelity/snap_tier_owned_envelope_20261002/verified_timing.json),
  [`independent quality`](results/baseline_fidelity/snap_tier_envelope_fresh_quality_20261002/verified_report.json).
- [ ] **P1 — Validate the write mechanism physically.** Count physical
  read/write pages, controller amplification and energy under longer
  workloads. The current logical reduction does not prove endurance or
  lifetime improvement.
- [ ] **P1 — Establish robustness and generality.** Test document-disjoint
  QA, summarization, short/long contexts, larger decode lengths, second
  seed/model, concurrency, slow-tier saturation, and regression cases.
  Sixteen exact-ID GPU/CPU pairs now pass integrity: empty generation no-op,
  one-token model, 1023/1024/1025 hot boundary, 3071/3072/3073 total-budget
  boundary, 16K and maximum 31,500-token context, and depth-controlled needles.
  Full-fill cases reach 2048 cold rows in all 32 layers and bursts of 1024
  promotions / 1056 paid writes, with exact state, trajectories and predictions.
  **Retrieval quality fails:** both sparse policies miss all six rare-code
  prompts at 16K/max context with repeated similar-key decoys; full-cache
  answers all six. No layer retains the complete 12-token answer span;
  29–32 layers drop it entirely (all 32 at 10% depth). The batching backend
  preserves the GPU policy; their shared prefill retention remains deficient.
  Keep retrieval robustness and broader promotion open. Investigate this
  alongside the earlier Qasper losses before advancing placement defaults.
  Evidence: [`boundary/retrieval report`](results/hardware/batched_scores_boundary_retrieval_20261001/verified_report.json).
  A faithful upstream-selector SnapKV diagnostic now recovers all six
  identical needle inputs. Its actual selected indices retain complete answer
  spans in 429–594 of 1024 layer/query-head pairs; Tiered retains none in any
  whole layer. This strengthens the case for investigating prompt retention,
  but does not isolate ranking from head-specific storage/resources: SnapKV
  owns 524 MiB of growing expanded-head KV, versus Tiered's different fixed
  8-KV-head layout. No matched-resource or independent quality gate passed.
  Read-only observation preserves the original selector AST and bitwise CPU
  output; the initial observation-verifier failure is preserved separately.
  **Latest validation: 127 full-suite tests passed; one existing dependency warning.**
  Evidence: [`SnapKV needle diagnostic`](results/baseline_fidelity/snapkv_needles_verified_v2_20261001.json).
  A paired Tiered-only isolation now recovers all six with the existing
  actual-query prefill option, versus zero with K-as-Q. Initial hot+cold KV
  payload is identical (320 MiB), as are initial row/visibility budgets;
  the full answer span survives in 22–29 layers. Fresh controls reproduce
  every frozen prediction/state/trajectory/counter. This supports changing
  placement as the research direction without expanded-head storage, but
  the option adds prefill work and remains experimental: the earlier
  24-document holdout interval crossed zero and had severe regressions.
  Those held-out rows must not be reused for tuning. Default stays unchanged;
  independent quality and joint prefill/decode/resource gates remain open.
  Evidence: [`actual-query isolation`](results/baseline_fidelity/tiered_query_needles_verified_v2_20261001.json).
  Integration with the batched CPU+score backend now preserves all six
  actual-query GPU outputs exactly: final complete cache/metrics, token IDs,
  trajectories, retention, row/byte geometry and counters. Each answer is
  recovered; physical reservation remains 384.125 MiB host and transfer
  payload differs from GPU storage. This is integration integrity, not new
  independent quality or speed evidence.
  Evidence: [`CPU actual-query integration`](results/baseline_fidelity/cpu_query_needles_verified_20261001.json).
  A conservative corpus audit excludes recorded sample IDs, prior manifest
  selections/prefixes and normalized context matches across standard/E data.
  It found **zero unused NarrativeQA documents** in the current 20-document
  slice. A new **100-document / five-task** tranche is frozen and verified
  (20 each Qasper, MultiFieldQA, HotpotQA, TriviaQA, GovReport); its three-arm
  evaluation is complete (300 arms), independently verified. The protocol compares GPU proxy, GPU actual-query
  and CPU actual-query, with paired document quality and exact CPU/GPU
  integrity. NarrativeQA requires additional unseen documents; semantic near
  duplicates/training overlap remain unverified. Do not relabel old documents
  as independent or use this frozen tranche for subsequent tuning.
  Evidence: [`fresh-document manifest verification`](results/baseline_fidelity/realq_cpu_fresh_documents_20261001/verified_manifest.json).
  **Verified fresh-document quality:** CPU actual-query minus GPU proxy is
  **+5.87 task-balanced points, paired document 95% CI [3.16, 8.83]**.
  The preregistered NI margin −0.5 and aggregate superiority gates pass.
  GPU/CPU actual-query match every generated ID, score, final complete state,
  counter and selection trajectory across all 100 documents. Per-task gains:
  Qasper +12.15, MultiFieldQA +8.27, HotpotQA +5.93, TriviaQA +2.50,
  GovReport +0.48. **17 documents regress; five lose >10 points.**
  This is conditional on five tasks/one model/seed and equal configured KV
  budgets, with unequal physical resources; it is not a faithful-baseline,
  uniform per-example, latency, NarrativeQA, or SOTA claim. Do not tune on this
  now-used independent tranche. The original process completed normally and
  GPU0 was freed before the next job.
  Evidence: [`fresh-document quality report`](results/baseline_fidelity/realq_cpu_fresh_documents_20261001/verified_report.json).
  **Verified combined-path systems gate:** actual-query GPU versus batched
  CPU+score passed full 4K/8K/16K five-step boundary-state parity. Three paired
  16K/50-step blocks give **81.33 / 75.27 ms/token**, **7.45% faster CPU**,
  p95 89.45 / 84.70 ms; every block improves. TTFT 2179.57 / 2254.99 ms
  (+3.46%) passes the 10% guard. Final GPU allocation is about 223 MB lower;
  host occupied/reserved is 211.81 / 402.78 MB. Submitted H2D/D2H payload is
  315.62 / 211.81 MB including initial writes. This is same-placement backend
  evidence on one fixed stream, with unequal host resources, not a ≥10%
  whole-model memory reduction, SnapKV speed, concurrency or STT result.
  Source/raw/independent equations are frozen; both GPU jobs completed normally.
  Evidence: [`combined-path systems gate`](results/hardware/realq_cpu_systems_20261001/verified_gate_report.json).
  The official NarrativeQA metadata is now frozen at commit
  `904246f6d1fe99a99a08a03501fe3e619af2cee5`. All 200 old questions map
  unambiguously to 20 validation-set story IDs. Excluding their IDs and
  normalized titles leaves all 355 official test-set stories available in
  metadata; 37 movies / 23 books have recorded 5K–25K word lengths.
  Original-story downloading, current content/cleanup integrity, actual token
  lengths and a separately frozen evaluation remain to be done. This is
  a verified route to new data, not independent quality evidence yet.
  Evidence: [`official story inventory`](results/baseline_fidelity/narrativeqa_official_story_inventory_verified_20261001.json).
  A separately frozen preflight selected 10 books + 10 movie scripts from
  unseen official test story IDs, one seeded QA/story, before download.
  **14/20 current-source downloads pass** bounded fetch, decoding, ordered
  official marker and 5K-word checks (10 books / 4 scripts). Six failures
  remain explicit: three charset conflicts, one unavailable source and two
  missing start markers. No story was replaced; no narrative model quality
  run has started. Successful contexts span 13,961–52,234 tokens, so some
  require documented standard truncation at 31,500. Current source bytes may
  differ from 2017; metadata/content checks do not prove semantic integrity.
  Next diagnose header-vs-body encodings from preserved raw bytes, then freeze
  the usable evaluation coverage without hiding any failures.
  Evidence: [`story preflight`](results/baseline_fidelity/narrativeqa_test_stories_20261001/verified_preflight.json).
  A separately preserved offline encoding adapter now raises readiness to
  **16/20** (10 books / 6 scripts), using the documents' explicit Latin-1
  body labels and strict Windows-1252 byte round-trips. It uses no new fetches
  or replacement IDs, and preserves the original failure report. Four stories
  remain flagged: one unavailable URL and three absent start markers. Those
  start markers are website JavaScript boilerplate (`if ( window`), which the
  visible-text extractor deliberately removes; this needs fidelity review,
  rather than feeding website code to the model merely to pass a check.
  At encoding preflight no NarrativeQA model evaluation had started; the
  subsequent16-story study now completed (top of page). Encoding evidence:
  [`encoding adapter verification`](results/baseline_fidelity/narrativeqa_test_stories_20261001/verified_encoding_v2.json).
- [ ] **P1 — Finish novelty and prior-art comparison.** FlexiCache already
  keeps host copies and offloads stable-head KV pages once, so saving repeat
  writes alone is not new. Compare against it and IceCache at controlled
  quality, latency, memory and transfer cost; test whether an asymmetric-write
  policy adds a real benefit. IceCache's bundled standard LongBench rows
  match our standard files, but our corrected control actually uses
  LongBench-E for five tasks; those requests differ. Its two prompt templates
  and default KV number format also differ. IceCache's separate bf16 branch
  compiled for this L40S and twice reproduced one corrected-control request
  with identical prompt token IDs and bf16 model/GPU/CPU cache values, but
  actual live memory use and attention work still differ; no fair multi-request
  comparison has passed yet. A one-request end snapshot now separates its
  **668 active GPU K/V pages** from **2,220 prefill pages awaiting release**;
  its pinned CPU pool held **3,581 K/V pages**. Each page is 64 KiB. These
  numbers are not peak memory or a matched TieredKV budget. See
  [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md).
  A separate 15,789-token stress row failed with IceCache's released 1 GiB
  GPU pool; increasing only its pool allocation to 4 GiB let it finish.
  That run reached 20.839 GB peak GPU allocation and still held 29,130
  deferred prompt pages at the end. It was outside our frozen 25-row control,
  so it is a feasibility result, not a quality comparison.
  On that **same prompt**, TieredKV scored 50/100 and omitted two of five
  requested names while IceCache scored 100/100. Their cache sizes differ, so
  this is a concrete TieredKV failure case, not a fair ranking. A prefill trace
  showed all 35 tokens in the source sentence with the five names were
  dropped from TieredKV's cache in all 32 layers; later promotion cannot
  recover them. In a fixed-capacity diagnostic, forcing those known tokens
  to stay changed the answer from three names (50/100) to all five
  (100/100). This is an answer-informed upper bound on one case, **not** a
  deployable fix. We still need a selector that works without the answer
  and survives other tasks' regression checks. A cheap question-word
  128-token window selector failed its first gate: it retained only 12 of
  the 35 source tokens, so we did not run it through the model. A sentence-
  based replacement gained 8.84 points in its first 24-question pilot, but
  only 1.22 points in a second independent 24-question set; the second
  uncertainty interval included a loss. Its replication gate failed, so
  it remains experimental and the default cache policy is unchanged.
  FlexiCache's unchanged native engine now builds and imports on L40S with
  torch 2.6.0+cu124 and transformers 4.50.0; the source worktree remains clean.
  The first model pilot failed before loading: its ctypes lookup selected
  host CUDA 13, which this driver cannot run. An explicit CUDA 12 runtime
  returns valid stream priorities. The resolver-only retry initialized the
  engine and allocated bf16 pools (24.71 GB GPU K/V, 193.27 GB pinned CPU
  K/V, 5.35 GB GPU minmax), but generation failed: its released prefill
  kernel requests 163,840 shared bytes versus L40S's 101,376-byte limit.
  A one-stage launch screen uses 65,536 shared bytes and passes causal GQA
  attention tolerances on three cases; fresh model attempts explicitly record
  both portability adapters. Two adapted 1,440-token requests completed with
  identical 64-token output and score 10/100. Pools, transfers and scheduled-
  token allocation traces match; idle poll counts differ (3 versus 7).
  Process cleanup is verified. D2H/H2D submitted payload was
  146.28/42.94 MB. These are feasibility observations at unequal resources,
  not a fair quality/speed comparison. Evidence:
  [`pair validation`](results/baseline_fidelity/flexicache_matched_input_pair_report_20261001.json).
  The current local suite passes 98 tests,
  including resolver scope and invalid-runtime regression checks.
  Evidence: [`native import`](results/baseline_fidelity/flexicache_engine_import_20261001.json).
  Live setup details are in
  [`engine setup`](results/baseline_fidelity/flexicache_engine_setup_20261001.json).
  Record allocated pools, occupied KV and attention work before expanding
  to a resource-matched comparison. Transfer tests alone do not validate
  the serving engine or establish speed, quality or write savings.
- [ ] **P2 — Final paper and reproducibility audit.** Freeze a supported
  claim, independent headline-table rerun, source/data hashes, figures,
  limitations, and an anonymized manuscript. The October venue plan is
  conditional on the earlier gates; research continues if they fail.

## Where to look when you need a specific answer

1. **Learn the basics:** [`SIMPLE_NOTES.md`](SIMPLE_NOTES.md) explains the
   terms, the token lifecycle and the main results in plain language.
2. **Status and next action:** this page. Update it after a milestone, not
   after every exploratory run.
3. **How the code works:** [`explanation.md`](explanation.md), then
   [`README.md`](README.md) for setup and commands.
4. **Evidence and decisions:** [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md)
   gives detailed analysis; [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) is the
   chronological ledger; raw JSON in `results/` is authoritative.
5. **Work controls:** [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md)
   defines advancement gates; [`REMEDIATION_CHECKLIST.md`](REMEDIATION_CHECKLIST.md)
   records the original review fixes and their remaining dependencies.
6. **Writing and dates:** [`PAPER_WORKING_DRAFT.md`](PAPER_WORKING_DRAFT.md)
   is the manuscript scaffold; [`SUBMISSION_SPRINT.md`](SUBMISSION_SPRINT.md)
   is a conditional schedule.

`TieredKV_Project_Explanation.md`, `AUDIT_VERIFICATION.md`, and
`TIEREDKV_INDEPENDENT_AUDIT.md` are older explanation/audit records. Keep them
for provenance; use `SIMPLE_NOTES.md`, `explanation.md` and this page for the
current state. Do not infer current performance claims from old figures or
documents without checking the linked raw results.
