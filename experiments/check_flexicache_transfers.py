"""Run the released FlexiCache transfer kernels; no vLLM/model import needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.utils.cpp_extension import load

from provenance import atomic_json, runtime_identity


COMMIT = "ab0f495b10fc6509c1637af35779b29121fb7d3f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(d2h, h2d, dtype):
    # Each layer owns 16 physical pages; each request owns eight of those.
    table = torch.arange(16, device="cuda", dtype=torch.int32).reshape(2, 1, 2, 4)
    source_table = table.expand(2, 2, 2, 4).contiguous()
    dest_table = 15 - source_table
    dest_table[:, :, 1, :] = -1  # unstable heads never offload
    offsets = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    original = torch.arange(2 * 32 * 16 * 128, dtype=torch.float32).reshape(2, 32, 16, 128)
    original = (original % 251).to(dtype)
    gpu = original.cuda()
    host = torch.full_like(original, -1, pin_memory=True)
    expected_host = torch.full_like(original, -1)
    records = []
    # Completed pages advance monotonically; already offloaded pages are excluded.
    for last_values, total_values in (([0, 1], [2, 3]), ([2, 3], [4, 4]), ([4, 4], [4, 4])):
        last = torch.tensor(last_values, device="cuda", dtype=torch.int32)
        total = torch.tensor(total_values, device="cuda", dtype=torch.int32)
        src, dst, count = d2h.build_d2h_pairs(
            source_table, dest_table, requests, last, total, offsets, offsets)
        expected_pairs = sorted(
            (layer * 16 + request * 8 + page,
             layer * 16 + 15 - (request * 8 + page))
            for request in range(2) for layer in range(2)
            for page in range(last_values[request], total_values[request]))
        actual_pairs = sorted(zip(src.cpu().tolist(), dst.cpu().tolist()))
        assert count == len(expected_pairs) and actual_pairs == expected_pairs
        # Match the released main/background stream handoff and wait for completion.
        ready = torch.cuda.Event()
        ready.record()
        background = torch.cuda.Stream()
        background.wait_event(ready)
        src.record_stream(background)
        dst.record_stream(background)
        with torch.cuda.stream(background):
            d2h.kv_cache_d2h_transfer(gpu, host, src, dst, 2, 2)
        background.synchronize()
        for source, dest in expected_pairs:
            expected_host[:, dest] = original[:, source]
        assert torch.equal(host, expected_host), "D2H changed K/V or untouched pages"

        restored = torch.zeros_like(gpu)
        h2d.register_cpu_kv_cache_pinned(host)
        ready.record()
        background.wait_event(ready)
        with torch.cuda.stream(background):
            h2d.kv_cache_h2d_transfer(host, restored, dst, src, 2, 2)
        background.synchronize()
        expected_gpu = torch.zeros_like(original)
        for source, _ in expected_pairs:
            expected_gpu[:, source] = original[:, source]
        assert torch.equal(restored.cpu(), expected_gpu), "H2D changed K/V or untouched pages"
        records.append({"last_completed_pages": last_values,
                        "total_completed_pages": total_values,
                        "pairs": actual_pairs, "pair_count": count,
                        "kv_payload_bytes_each_direction": count * 2 * 16 * 128 * original.element_size()})
    return {"dtype": str(dtype), "exact_kv_and_untouched_page_parity": True,
            "ranges": records, "pinned_buffer_bytes": host.numel() * host.element_size()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/data/nishant/tieredkv_external/flexicache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior evidence: {args.output}")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Expected frozen upstream {COMMIT}, got {revision}")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Expected an unchanged upstream checkout")
    capability = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    os.environ.setdefault("MAX_JOBS", "2")
    paths = {"d2h": args.upstream / "vllm/v1/flexicache/kernels/cuda/kv_d2h_tx_gpu_mapped.cu",
             "h2d": args.upstream / "vllm/v1/flexicache/kernels/cuda/h2d_copy_blocks.cu"}
    report = {"scope": "released transfer-kernel feasibility only; no model/vLLM or performance result",
              "upstream_commit": revision, "source_sha256": {name: sha(path) for name, path in paths.items()},
              "runner_sha256": sha(__file__), "runtime": runtime_identity(),
              "gpu": torch.cuda.get_device_name(), "compute_capability": list(capability),
              "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
              "checks": [], "passed": False}
    try:
        modules = {}
        for name, path in paths.items():
            directory = args.build_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            modules[name] = load(name=f"flexicache_check_{name}", sources=[str(path)],
                                 build_directory=str(directory), verbose=True)
        report["binary_sha256"] = {name: sha(module.__file__) for name, module in modules.items()}
        for dtype in (torch.float16, torch.bfloat16):
            report["checks"].append(check(modules["d2h"], modules["h2d"], dtype))
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
