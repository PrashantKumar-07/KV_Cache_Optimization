# Third-party sources

This repository's license does not replace upstream model, data or software
licenses. The bootstrap script downloads original sources at the pinned commits
listed in that script, retaining their own license files.

- **LongBench**, THU-KEG / Zhipu AI: the standalone scoring functions in
  `experiments/longbench_metrics.py` derive from the official evaluation code.
  Its MIT notice is retained in `LICENSES/LongBench.txt`.
  Source: <https://github.com/THUDM/LongBench>. The downloaded benchmark combines
  datasets with their own terms; consult the official dataset documentation.
- **SnapKV**, FasterDecoding: official prompt selector and reference prompts are
  loaded from the pinned checkout. Apache-2.0 notice:
  `LICENSES/SnapKV.txt`. Source: <https://github.com/FasterDecoding/SnapKV>.
- **KVCache-Factory**, Zefan Cai: the current GQA SnapKV selector is loaded from
  the pinned checkout. MIT notice: `LICENSES/KVCache-Factory.txt`.
  Source: <https://github.com/Zefan-Cai/KVCache-Factory>.
- Optional source checkouts are **IceCache** and **M-DCI** (Yuzhen Mao),
  **RocketKV** (NVIDIA), and **RAPIDS-CMake** (RAPIDS). Their licenses remain in
  the fetched repositories; their code is not relicensed by this project.
- The model helper downloads **Mistral-7B-Instruct-v0.2** from Mistral AI.
  Weights are not redistributed in this repository. Model card and license:
  <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>.

Historical source snapshots under `results/` also contain frozen upstream
H2O, Quest and StreamingLLM reference source, including their vendored projects.
Original available license/notice texts are preserved byte-for-byte in
`LICENSES/historical_source_licenses.tar.gz`; its adjacent JSON records paths
and checksums. These sources retain their upstream terms and are not relicensed
by this project. This archival material is audit evidence; use the documented
pinned bootstrap for active baselines.
