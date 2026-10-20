# TieredKV: start here

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

**Current status (3 October):** batchedGPU native read indices preserve
query behavior and pass **283tests**, native3dtype/stream/pool/cleanup and
547.488MiB/576pilot resource checks. Component3ABBA median improves24.61%,
but freshordinaryLRU guard fails1.810×median/1.679×p95. **Stop standalone
full-query expansion/adoption.** KVreads5.826GB plus75.33MBcandidate metadata
charged; no totalH2D saving claim. Next inspect fusedranking/lower-traffic
retrieval/GPU selected-row indices. Ordinary source/CLI/defaults unchanged;
full512MiB/actualmatching and P0/P1/P2 remain open. Estimate10–20 working days,
conditional on hardware/research. [Current evidence](PROJECT_STATUS.md).

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

**Earlier verified progress:** optional bulk promotion planning preserves the
model's state and outputs. At 16K it improves the same CPU backend by
**5.23% / 6.54%** at two cache capacities. Generation, lifetime, recycling and
ordinary evaluator checks pass; **206 tests passed at that milestone**. Add `--cpu-bulk-promotion`
to the existing native-reuse CPU options. Defaults stay unchanged.
Complete memory, quality, SnapKV speed and physical hardware goals remain open.
Estimate **10–20 working days**, conditional on research outcomes and hardware
access. [Current result and remaining work](PROJECT_STATUS.md).

Read this page first. It explains the project without assuming that you know
how a language model stores its memory. The technical documents are linked at
the end; you do not need to read them all at once.

## What problem are we trying to solve?

A language model writes an answer one small text piece at a time. Those pieces
are called **tokens**. To use an earlier token again, the model saves two sets
of numbers for it: a **key** and a **value**. Together, the saved keys and values
are the **KV cache**.

Long prompts create a large KV cache. The graphics card has limited memory.
TieredKV asks: *Can we keep a smaller useful cache on the graphics card, while
avoiding unnecessary writes when saved tokens move between memory groups?*

## The idea in one example

Imagine that the fast group can hold three saved tokens, and the slower group
can hold two. These sizes are just for the example.

| Moment | Fast group | Slower group | What happened |
| --- | --- | --- | --- |
| After reading the prompt | A, B, C | D, E | The program chose which old tokens to keep. |
| D becomes useful | A, C, D | B, E | D moved up; B made room by moving down. |
| D later becomes less useful | A, B, C | D, E | D can move down again. |

If the program kept an unchanged backup of D in the slower group, it can reuse
that backup during the last move. It does not need to write D there again.
That is the **write-saving idea**. A token that was discarded completely cannot
be recovered by this program.

There is an important catch: the main experiments put **both groups on the
graphics card**. They test the decision rules and count logical writes; they
do not measure a real STT-RAM device. An opt-in backend now puts the slower group in ordinary computer memory
and batches its transfers. It is faster than the GPU version with the same
placement, but still slower than SnapKV in the corrected comparison. STT-RAM is the proposed future memory technology, not hardware used in
the current experiments.

## What have we shown so far?

1. **The program runs, and the backup idea saves repeat writes in controlled
   tests.** On one fixed 50-step test, it counted about 90% fewer writes to
   the slower group. A separate ordinary-computer-memory test also transferred
   less data. Neither test measures STT-RAM chip lifetime.
2. **We have not shown better answers than a strong comparison method.** In
   the current 135-question comparison with SnapKV, the score difference was
   too uncertain to declare a winner. A newer CPU-backed version, tested on
   100 unused documents with shared cache limits, scored about 2.6 points
   lower than SnapKV. It has not replaced the default.
3. **The current version is slower in one measured long-prompt test.** It took
   about 83 milliseconds per generated token; SnapKV took about 26. A newer
   CPU-backed comparison with the same maximum cache-memory and attention
   limits took about 71 versus 26 milliseconds. Actual usage still differs.
   Smaller time is better.
4. **A quality failure is understood, but not fixed.** On one question asking
   for five names, TieredKV discarded the sentence containing the names. It
   answered only three. An experimental search method helped on a first set
   of questions but did not reliably help on a second set, so it has not
   replaced the main method.

These results show a useful mechanism and clear research gaps. They do not yet
show that TieredKV is better overall than published methods.

## Words you will see in the other files

| Word | Meaning in this project |
| --- | --- |
| Graphics card / GPU | The processor that runs most model calculations. |
| Graphics-card memory / VRAM | Fast memory attached to that processor. |
| CPU memory | Ordinary computer memory; used as the slower group only in an optional experiment. |
| STT-RAM | A proposed memory technology. We have not tested a physical device. |
| Attention | The model's way of using earlier saved tokens when writing the next one. |
| Promotion | Moving a saved token from the slower group to the fast group. |
| Demotion | Moving a saved token from the fast group to the slower group. |
| Eviction | Discarding a token when there is no room; the model cannot fetch it back. |
| Shadow / backup | An unchanged extra copy kept to avoid writing the same token again. |
| Baseline | Another method tested on the same questions for comparison. |
| Latency | How long an operation takes; lower is faster. |
| Throughput | How much work finishes in a period of time; higher is better. |
| SOTA | “State of the art”: the strongest published methods. We have not proved we beat them. |

## What to read next

1. [PROJECT_STATUS.md](PROJECT_STATUS.md): what is finished and what remains.
   Read only “Current state” and “Remaining work” on a first pass.
2. [SIMPLE_NOTES.md](SIMPLE_NOTES.md): a longer plain-language walkthrough and
   definitions for more words.
3. [explanation.md](explanation.md): the exact code behavior. Use it as a
   reference when you need to answer a specific technical question.

The [experiment log](EXPERIMENT_LOG.md) and [research roadmap](RESEARCH_ROADMAP.md)
are detailed evidence records. You do not need to study them before you can
explain the project to your supervisor.

To use the verified native prefill option, add `--prefill-last-logits` to an
existing LongBench, query/bulk, latency-profile or fresh-memory command. The
model must explicitly support `logits_to_keep`; the locally tested Mistral does.
Python callers can pass `prefill_last_logits=True` to `generate_with_policy`,
`evaluate_task` or `profile_one`. Use a separate result path for changed options.
The memory saving concerns the declared decode scope; prefill peak and complete
512 MiB resource equality remain open.
