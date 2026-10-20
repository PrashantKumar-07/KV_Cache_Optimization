# Cross-layer eviction projections: failed charged screen

Research coordinator replaces per-layer CUDA cumulative/age projection tensors
with batched pad/add/mask operations. The original stable victim selection,
grace/hard masks, tie order and fallback logic are reconstructed from its AST;
only projected score preparation changes. Ordinary bulk cache class, retention,
source, CLI and defaults are unchanged. The experimental class has no CLI wiring.

Preregistered `plan.json` and control source archive precede implementation.
19 focused CPU tests and all347tests pass. Native CUDA bytecode was not changed.
Six GPU cases check exact victim lists and ragged/nonfinite/grace projection
bits across1/4/32 appends. Charged preparation+selection includes CPUpositions,
grace concatenation, CUDA allocation/padding/age/masks/copies, stable sort,
finite counts, CPUreturn and fallback. Promotion/model/physicalwrites are excluded.

ThreeABBA,20 measured calls after4warmups per row,25before/after quietGPU0 checks:
median3.419→2.686ms (0.733ms saved), p953.502→5.831ms. Allblock medians improve,
but required4ms saving and <=2%p95regression both fail. `verify_stage.py`
independently checks frozen source/archive/tests/row counts/numeric gates.
**Stop projection-only expansion**. No fullmodel/resource/timing/generation run
is justified. This screen does not prove model preservation, whole-loop gain,
complete512MiB, independentquality or physicalSTT. No favorable timing retry.

All owned jobs are terminal. See `verified_stage.json`, `stage_summary.csv`,
original `stage.json`, source archives, CPUtestlogs and `status_update.json`.
Next target broader ordinary layer-state transitions/synchronization under a
new preregistered complete-mechanism gate. FullP0/P1/P2 remains open; the
unsupported repeated10–20workingdays forecast stays withdrawn.
