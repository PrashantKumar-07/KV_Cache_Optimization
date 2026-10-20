# Exact key scan and selected-value reads

Research-only Mistral reader: scan all existing cold keys with unchanged
query/rotary/ranking/top-k, then gather only the selected cold values. The native
kernel copies a single channel from the same pinned pool using validated GPU
source maps. Visible hot prefixes, destination ownership and strides are retained.
Ordinary source, CLI and defaults were not changed by this mechanism.

Evidence is staged, preregistered, and hashed:

1. `plan.json`, source archives, 22 focused tests and 328 full-suite tests.
2. `native_v2.json`: BF16/FP16/F32 bitwise copies, 36 metadata guards,
   empty reads, serial stream handoff, alternating pools and pinned-root cleanup.
   Numerical GPU indices originate from validated maps/top-k; invalid numerical
   indices were not deliberately dispatched into the shared CUDA context.
3. `gpu.json`, `transfer_accounting.json`, `verified_correctness.json`:
   exact 12 predictions, 13 logit/cache/counter frames and 384 visible selections.
   Both arms retain destinations and canonical hot prefixes.
4. `resource.json`, `verified_resource.json`: fresh 16K/50-step replay, exact
   predictions/state/counters; copied native K/V H2D 5,825,961,984 to
   3,173,392,384 bytes (45.53% reduction). Unchanged native D2H 115,855,360 bytes,
   index-copy metadata 75,333,120 bytes and 51 index refreshes. H2D calls increase
   1,701 to 3,301; key/value reads each 1,600. Active decode remains 545.17 MiB,
   above 512 MiB but within the declared 576 MiB research pilot. Rounded host
   storage is 256 MiB, whole reserved envelope 5,089.49 MiB, prefill peak 18.62783 GB.
5. `cost_plan.json`, `cost_source.tar.gz`, `cost_implementation.json` freeze the
   paired-reader versus channel-reader 3ABBA comparison before execution.
   Primary: at least 5% median improvement and 4 ms saved; p95 at most 2% worse,
   TTFT at most 10% worse. Passing requires a separate fresh ordinary-LRU gate.
   A failed primary stops standalone expansion/adoption.

The first native build failed before kernel dispatch because Ninja was absent
from the shell PATH. `native_build_plan_v2.json` records the existing Ninja and
CUDA 12.4 toolchain used without installing dependencies or changing the kernel.
The original failed report, source and log remain preserved. Raw GPU stdout
inherited an inaccurate "exact KV bytes" label; the separate scope correction
keeps raw data intact and identifies intentionally changed H2D accounting.

All native/model jobs use GPU0 and the shared GPU lock. The measured invocation
uses the existing toolchain environment:

```bash
PATH=/data/nishant/tieredkv_external/flexicache_venv/bin:/usr/local/cuda-12.4/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=0 \
flock /data/nishant/tieredkv_external/gpu0_model.lock \
../venv_kvcache/bin/python results/hardware/query_channel_reads_v1/check_cost.py
```

Drivers refuse to overwrite raw evidence; reproduction needs a separate output
location and new provenance. The measured binary SHA256 is
`4194dcd812f5a3dd43410647a3bb0ed27fbb447b371a9af0a5ae98f237858461`.
Source archives include Python and native C++/CUDA/header files.

Copied K/V payload excludes library transfers and physical PCIe/STT reads.
This is pinned CPU DRAM, not physical STT. One model/seed teacher replay does
not establish independent task quality, parallel serving, matched faithful
baseline performance, a complete 512 MiB envelope, energy or endurance.

Final disposition: quiet3ABBA97.48→93.02ms, **4.57% / 4.46ms saved**.
P95100.49→95.90ms; TTFT+0.05%. The required >=5% median gain fails;
**stop standalone channel-reader expansion/adoption**. No ordinaryLRU or wider
quality run follows this failed gate. All owned jobs are terminal. See
`verified_cost.json`, `channel_tradeoff.pdf` and `verified_status.json`.
