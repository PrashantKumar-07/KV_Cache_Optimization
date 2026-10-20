# TieredKV

Research code for a bounded hot KV cache backed by a recoverable cold tier for LLM inference. STT-RAM traffic and energy are simulated; the optional CPU-offload implementation is a separate measured system.

**Current status:** the audit and improvement work remain in progress. No SOTA superiority or publication readiness is established. See [the current checkpoint](START_HERE.md) before continuing experiments.

## Run from a fresh clone

Validated setup: Linux x86_64, Python 3.10, PyTorch 2.5.1 CPU, Transformers 5.15.0. Installation requires internet; the following tests and smoke run need no model download or GPU.

```bash
git clone https://github.com/PrashantKumar-07/KV_Cache_Optimization.git
cd KV_Cache_Optimization
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.cpu.lock.txt
python scripts/restore_evidence.py
python scripts/bootstrap_references.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python scripts/smoke.py
```

The smoke command performs actual simulated-cache prefill and 16 decode steps, checking finite attention, capacity and write accounting. The core tests include tiny randomly initialized Mistral models. See [REPRODUCING.md](REPRODUCING.md) for the 14-case official GQA-selector tests, pinned model/data download, CUDA setup and an explicit one-record evaluation command.

Full Mistral 7B evaluations require model access, downloaded weights/data and sufficient GPU memory. The recorded main experiments used NVIDIA L40S 48 GB GPUs. Optional upstream IceCache/RocketKV environments have outstanding correctness gates; they are not certified by the CPU setup tests.

## Results and figures

- [Complete seven-configuration evaluation](results/full_evaluation_20261015/verified_results.json): 10,318 real predictions across 1,474 records, independently checked.
- [Official current GQA SnapKV evaluation](results/gqa_baseline_20261016/verified_results.json): 1,474 independently rescored predictions; task macro 39.523. The existing Tiered hot1024 macro is 38.809. This is descriptive consumed-data evidence with different cache resources, not a fair-memory SOTA win.
- [Figure status and provenance](figures_final/README.md): figures 1/2/3/4/6/7/11/12/14 use matching verified runs. Figures 8/15 remain provisional. Original chart questions/styles are preserved; adapted functional controls are explicitly distinguished from official baselines.
- [Editable architecture](figures/tieredkv_architecture_paper.svg) and [implementation/methodology](docs/METHODOLOGY.md).

Figure 3 counts avoided decode payload writes. Including initial cold placement yields 40.6793% pooled whole-request logical avoidance for its hot992 configuration, 26.0881% equal-task macro. This is fixed-trace accounting, not an executed exclusive-policy ablation or physical endurance measurement. Broader modeled traffic/energy must not be called measured device performance.

## Continue the research

[START_HERE.md](START_HERE.md) is the single human-facing checkpoint; [TASK_STATE.json](TASK_STATE.json) stores the same current state for automation. It records completed runs, failures, consumed/reserved data, the next work and paper gates. Old process IDs are historical; check real jobs before launching anything. Paper drafting follows audit, fair results and figure validation.

Raw positive/negative evidence and frozen plans remain under `results/`. Oversized raw JSON is preserved byte-for-byte in checksummed gzip parts under `artifacts/evidence/`; the restore command reconstructs the original paths. Virtual environments, downloaded model/data copies, upstream clones and compiled binaries are rebuilt from documented pinned sources. Superseded prose/checklists and the old paper draft are preserved in one [historical archive](docs/archive/manifest.json), outside the active documentation.

[License](LICENSE) · [Third-party attribution](THIRD_PARTY_NOTICES.md)
