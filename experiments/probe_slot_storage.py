"""Isolated packed-compaction versus recycled-slot KV storage probe.

This is a data-movement screen, not an end-to-end TieredKV benchmark.
"""
import argparse
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


def run_arm(kind, initial, incoming, victims, reads, device):
    states = [(k.clone(), v.clone()) for k, v in initial]
    logical_to_slot = list(range(initial[0][0].shape[1]))
    samples = []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    for step, (drop, selected) in enumerate(zip(victims, reads)):
        start = time.perf_counter()
        old_map = logical_to_slot
        dead = set(drop)
        survivors = [i for i in range(len(old_map)) if i not in dead]
        if kind == "packed":
            keep = torch.tensor(survivors + list(range(len(old_map), len(old_map) + len(drop))), device=device)
            read_idx = torch.tensor(selected, device=device)
        else:
            freed = [old_map[i] for i in drop]
            logical_to_slot = [old_map[i] for i in survivors] + freed
            write_idx = torch.tensor(freed, device=device)
            read_idx = torch.tensor([logical_to_slot[i] for i in selected], device=device)
        for layer, (k, v) in enumerate(states):
            new_k, new_v = incoming[step][layer]
            if kind == "packed":
                k = torch.cat((k, new_k), dim=1).index_select(1, keep)
                v = torch.cat((v, new_v), dim=1).index_select(1, keep)
                states[layer] = (k, v)
            else:
                k.index_copy_(1, write_idx, new_k)
                v.index_copy_(1, write_idx, new_v)
            # Same 32 logical rows read in both arms; physical layout may differ.
            k.index_select(1, read_idx)
            v.index_select(1, read_idx)
        torch.cuda.synchronize(device)
        samples.append(1000 * (time.perf_counter() - start))
    peak = torch.cuda.max_memory_allocated(device)
    canonical = [(k.index_select(1, torch.tensor(logical_to_slot, device=device)).cpu(),
                  v.index_select(1, torch.tensor(logical_to_slot, device=device)).cpu())
                 for k, v in states] if kind == "slots" else [(k.cpu(), v.cpu()) for k, v in states]
    del states
    return {"step_ms": samples, "peak_bytes": peak}, canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--new-rows", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--output", default="results/decode_overhead/slot_storage_probe.json")
    args = parser.parse_args()
    if args.rows < args.new_rows or args.new_rows < 1:
        parser.error("require rows >= new-rows >= 1")
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(947)
    shape = (8, args.rows, 128)
    initial = [(torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator),
                torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator))
               for _ in range(args.layers)]
    incoming = [[(torch.randn((8, args.new_rows, 128), device=device, dtype=torch.bfloat16, generator=generator),
                  torch.randn((8, args.new_rows, 128), device=device, dtype=torch.bfloat16, generator=generator))
                 for _ in range(args.layers)] for _ in range(args.steps)]
    victims = [[(step * 37 + i * 3) % args.rows for i in range(args.new_rows)]
               for step in range(args.steps)]
    reads = [[(step * 41 + i * 47) % args.rows for i in range(min(32, args.rows))]
             for step in range(args.steps)]
    rows = []
    reference = None
    for block in range(args.blocks):
        for kind in ("packed", "slots", "slots", "packed"):
            stats, canonical = run_arm(kind, initial, incoming, victims, reads, device)
            if reference is None:
                reference = canonical
            elif not all(torch.equal(a, b) for pair, expected in zip(canonical, reference)
                         for a, b in zip(pair, expected)):
                raise AssertionError(f"K/V parity failure: {kind} block {block}")
            row = {"block": block, "arm": kind, **stats}
            rows.append(row)
            print({"block": block, "arm": kind,
                   "median_ms": statistics.median(stats["step_ms"]),
                   "peak_gb": stats["peak_bytes"] / 1e9}, flush=True)
            del canonical
            gc.collect()
            torch.cuda.empty_cache()
    summary = {kind: {"median_ms": statistics.median(t for r in rows if r["arm"] == kind for t in r["step_ms"]),
                      "p95_ms": sorted(t for r in rows if r["arm"] == kind for t in r["step_ms"])[int(.95 * (2 * args.blocks * args.steps - 1))],
                      "peak_gb": max(r["peak_bytes"] for r in rows if r["arm"] == kind) / 1e9}
               for kind in ("packed", "slots")}
    result = {"scope": "isolated full slow-tier K/V replacement plus 32-row read; no policy or model",
              "config": vars(args), "runtime": runtime_identity(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(device), "parity": "exact K/V after each arm",
              "rows": rows, "summary": summary,
              "limitations": "No promotion, attention, CPU/STT transfer, quality, or model forward. Fixed full capacity only."}
    atomic_json(args.output, result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
