# Reproduce and continue

Start from the repository root. The validated core environment is Linux x86_64,
Python 3.10, PyTorch 2.5.1 and Transformers 5.15.0. Git and Python's `venv`
module are required. Use several GB of free disk for the environment; model
experiments additionally require the downloaded weights and an appropriate GPU.

## Offline computation with no model or GPU

Installation and fetching the two pinned references require internet access.
The checks themselves download no models or datasets.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.cpu.lock.txt
python scripts/restore_evidence.py
python scripts/bootstrap_references.py
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python scripts/smoke.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=experiments python -m pytest -q results/improvement_design_20261016/test_gqa_snap_adapter.py
python experiments/longbench_eval.py --help
```

The core suite includes tiny randomly initialized Mistral integration tests.
A CUDA-specific check skips on CPU. `scripts/smoke.py` executes actual cache
prefill and 16 decode steps, checking finite outputs, tier capacity and write
accounting. Its traffic latency/energy are simulated estimates, not device
measurements or model quality. The separate 14-case selector test checks the
official GQA SnapKV compatibility bridge. Test logs and the exact fresh CPU
environment are recorded in the publication validation evidence.

`requirements.txt` pins the tested direct dependencies. `requirements.cpu.lock.txt`
pins the complete fresh CPU environment. `requirements.lock.txt` preserves the
historical server CUDA package inventory, including unrelated installed tools;
use the CPU lock above for a clean CPU installation.

## Model evaluation on CUDA

Create a separate environment. The recorded main runtime used CUDA 12.4 Torch
2.5.1 wheels and NVIDIA L40S 48 GB GPUs. A suitable driver, sufficient VRAM and
free host RAM/disk are required. Other hardware/backends need their own validation.

```bash
python3.10 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/restore_evidence.py
python scripts/bootstrap_references.py
python download_model.py --output models/mistral-7b-instruct
python scripts/download_longbench.py --output data/longbench
```

The model helper pins the original snapshot revision, uses portable paths, and
honors normal Hugging Face credentials/cache settings. Download access is subject
to the model provider's terms. The data helper pins the official LongBench archive
and verifies both its SHA-256 and the six exact JSONL hashes used by the evaluations.
Use `--archive /path/to/data.zip` to reuse an existing verified archive offline.

For a new one-record functional check, write to a **new** output directory:

```bash
mkdir -p results/local_smoke
CUDA_VISIBLE_DEVICES=0 python experiments/longbench_eval.py \
  --model models/mistral-7b-instruct --data-dir data/longbench \
  --tasks multifieldqa_en --methods full tieredkv --budget 1024 \
  --max-samples 1 --prefill-last-logits --tiered-kv-dtype bfloat16 \
  --json results/local_smoke/comparison.json
```

This example checks execution; it is not the registered full comparison and does
not establish SOTA superiority. The main CLI's simplified comparator policies
must not be described as official SOTA implementations. Read the current
checkpoint before selecting an experiment or interpreting results.

## Official research baselines and historical runs

`python scripts/bootstrap_references.py --all` also fetches pinned official
IceCache BF16, M-DCI BF16, RocketKV and RAPIDS-CMake sources. This fetches sources
only. Their separate package/compiler environments, layout correction, GPU
validation and resource-matched comparison gates remain recorded in the current
checkpoint; fetching does not resolve those gates or certify the baselines.
Do not install their legacy requirements into the main environment.

Historical `results/**/run.py`, build receipts and frozen plans intentionally
preserve original server paths, runtime identities and source hashes. They are
audit evidence, not portable launch commands: changing their paths/hashes silently
would invalidate provenance. Continue from `START_HERE.md` and `TASK_STATE.json`,
verify terminal/live job state, and register a new portable run with its own output
directory and current model/data/runtime identities. Preserve consumed-data and
failed-candidate restrictions. Restore archived evidence before running verifiers.

Model weights, local dataset copies, virtual environments, compiler build trees
and generated CUDA binaries are excluded from Git. Rebuild platform-specific
binaries in the correct isolated environment; never reuse server binaries as
proof on a different machine. See `THIRD_PARTY_NOTICES.md` for source attribution.
