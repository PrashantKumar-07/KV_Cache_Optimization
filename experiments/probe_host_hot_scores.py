"""Synthetic falsification screen for host hot-score bookkeeping, not model latency."""
import argparse
import hashlib
import statistics
import time
from pathlib import Path

import torch

from provenance import atomic_json, runtime_identity


def select(cumulative, age, attention, protected):
    updated = cumulative + attention
    older = age + 1
    scores = updated / older.clamp(min=1)
    scores[protected] = float("inf")
    order = torch.argsort(scores, stable=True)
    take = min(33, int(torch.isfinite(scores).sum().item()))
    return updated, older, scores, order[:take].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior evidence: {args.output}")
    torch.manual_seed(0)
    shape = (32, 1025)
    cumulative = torch.randint(0, 128, shape).float() / 128
    age = torch.randint(0, 64, shape).float()
    attention = torch.randint(0, 32, shape).float() / 32768
    attention[:, 992:] = 0
    protected = torch.zeros(shape, dtype=torch.bool)
    protected[:, :4] = True
    protected[:, -161:] = True
    # Include scarce ordinary victims and an all-protected boundary.
    protected[0, :] = False
    protected[0, :1000] = True
    protected[1, :] = True
    gpu_cumulative, gpu_age, gpu_attention = (
        tensor.cuda() for tensor in (cumulative, age, attention))
    exact = True
    for i in range(shape[0]):
        host = select(cumulative[i], age[i], attention[i], protected[i])
        device = select(gpu_cumulative[i], gpu_age[i], gpu_attention[i], protected[i])
        exact &= all(torch.equal(a, b.cpu()) for a, b in zip(host[:3], device[:3]))
        exact &= host[3] == device[3]

    def device_path():
        return [select(gpu_cumulative[i], gpu_age[i], gpu_attention[i], protected[i])[3]
                for i in range(shape[0])]

    def host_path():
        copied = gpu_attention.cpu()
        return [select(cumulative[i], age[i], copied[i], protected[i])[3]
                for i in range(shape[0])]

    timings = {"device": [], "host": []}
    for _ in range(2):
        device_path()
        host_path()
    for arm in ("device", "host", "host", "device") * 3:
        torch.cuda.synchronize()
        start = time.perf_counter()
        (device_path if arm == "device" else host_path)()
        torch.cuda.synchronize()
        timings[arm].append((time.perf_counter() - start) * 1000)
    medians = {arm: statistics.median(values) for arm, values in timings.items()}
    saved = medians["device"] - medians["host"]
    report = {"scope": "synthetic arithmetic only; excludes real cache updates and full-loop overhead",
              "config": {"runtime": runtime_identity(), "seed": 0, "shape": list(shape),
                         "device": torch.cuda.get_device_name(), "dtype": "float32",
                         "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "score_age_and_stable_victims_exact": bool(exact),
              "bulk_attention_copy_bytes": attention.numel() * attention.element_size(),
              "timings_ms": timings, "median_ms": medians, "saved_ms": saved,
              "screen_passed": bool(exact and saved >= 4)}
    atomic_json(args.output, report)
    print({key: report[key] for key in ("score_age_and_stable_victims_exact", "median_ms", "saved_ms", "screen_passed")})


if __name__ == "__main__":
    main()
