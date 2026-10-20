"""Screen reserved slow-tier append for the 75%-full decode regime."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import atomic_json, runtime_identity


def run(kind, initial, additions, reads, capacity, device):
    used = initial[0][0].shape[1]
    if kind == "reserved":
        state = []
        for k, v in initial:
            kb = torch.empty((k.shape[0], capacity, k.shape[2]), device=device, dtype=k.dtype)
            vb = torch.empty_like(kb)
            kb[:, :used, :].copy_(k)
            vb[:, :used, :].copy_(v)
            state.append((kb, vb))
    else:
        state = [(k.clone(), v.clone()) for k, v in initial]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for step, rows in enumerate(additions):
        start = time.perf_counter()
        n = rows[0][0].shape[1]
        for layer, (k, v) in enumerate(state):
            nk, nv = rows[layer]
            if kind == "reserved":
                k[:, used:used+n, :].copy_(nk)
                v[:, used:used+n, :].copy_(nv)
            else:
                k = torch.cat((k, nk), dim=1)
                v = torch.cat((v, nv), dim=1)
                state[layer] = (k, v)
            idx = reads[step]
            k.index_select(1, idx)
            v.index_select(1, idx)
        used += n
        torch.cuda.synchronize(device)
        samples.append(1000 * (time.perf_counter() - start))
    peak = torch.cuda.max_memory_allocated(device)
    canonical = [(k[:, :used, :].cpu(), v[:, :used, :].cpu()) for k, v in state]
    del state
    return {"step_ms": samples, "peak_bytes": peak}, canonical


def main():
    device = torch.device("cuda:1")
    layers, initial_rows, capacity, steps = 32, 1536, 2048, 16
    generator = torch.Generator(device=device).manual_seed(948)
    shape = (8, initial_rows, 128)
    initial = [(torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator),
                torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator))
               for _ in range(layers)]
    additions = [[(torch.randn((8, 1, 128), device=device, dtype=torch.bfloat16, generator=generator),
                   torch.randn((8, 1, 128), device=device, dtype=torch.bfloat16, generator=generator))
                  for _ in range(layers)] for _ in range(steps)]
    reads = [torch.tensor([(step * 41 + i * 47) % (initial_rows + step + 1)
                           for i in range(32)], device=device) for step in range(steps)]
    rows, reference = [], None
    for block in range(2):
        for kind in ("packed", "reserved", "reserved", "packed"):
            stats, canonical = run(kind, initial, additions, reads, capacity, device)
            if reference is None:
                reference = canonical
            elif not all(torch.equal(a, b) for pair, expected in zip(canonical, reference)
                         for a, b in zip(pair, expected)):
                raise AssertionError("K/V parity failure")
            rows.append({"block": block, "arm": kind, **stats})
            print(kind, block, statistics.median(stats["step_ms"]),
                  stats["peak_bytes"] / 1e9, flush=True)
            del canonical
            gc.collect()
            torch.cuda.empty_cache()
    summary = {kind: {"median_ms": statistics.median(t for r in rows if r["arm"] == kind
                                                    for t in r["step_ms"]),
                      "max_peak_gb": max(r["peak_bytes"] for r in rows if r["arm"] == kind) / 1e9}
               for kind in ("packed", "reserved")}
    result = {"scope": "isolated 32-layer 1536-to-1552-row append and 32-row read",
              "config": {"layers": layers, "initial_rows": initial_rows,
                         "capacity": capacity, "steps": steps,
                         "new_rows_per_step": 1, "device": str(device)},
              "runtime": runtime_identity(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(device),
              "parity": "exact final K/V", "rows": rows, "summary": summary,
              "limitations": "Reserved buffer is allocated before timing; peak includes it but excludes model/policy/attention."}
    atomic_json("results/decode_overhead/reserved_append_probe.json", result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
