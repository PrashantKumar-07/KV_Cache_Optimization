"""Charged cross-layer versus per-layer native pinned KV transfer screen."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from probe_cpu_slots import load_kernel
from provenance import atomic_json, runtime_identity


def arm(kind, initial, incoming, selections, capacity, d2h, h2d):
    layers, _, heads, rows, dim = initial.shape
    pool = torch.zeros((2, layers * heads * capacity, 1, dim),
                       dtype=initial.dtype, pin_memory=True)
    view = pool.reshape(2, layers, heads, capacity, dim)
    view[:, :, :, :rows].copy_(initial.permute(1, 0, 2, 3, 4))
    h2d.register_cpu_kv_cache_pinned(pool)
    samples, reads = [], []
    torch.cuda.synchronize()
    for step, chosen in enumerate(selections):
        width = len(chosen[0])
        start = time.perf_counter()
        groups = [list(range(layers))] if kind == 'batched' else [[i] for i in range(layers)]
        outputs = []
        for group in groups:
            packed = torch.stack([incoming[step][i] for i in group], dim=1)
            dst = torch.tensor([i * heads * capacity + h * capacity + rows + step
                                for i in group for h in range(heads)],
                               device='cuda', dtype=torch.int32)
            src = torch.arange(len(group) * heads, device='cuda', dtype=torch.int32)
            d2h.kv_cache_d2h_transfer(packed.reshape(2, -1, 1, dim), pool, src, dst)
            source = torch.tensor([i * heads * capacity + h * capacity + r
                                   for i in group for h in range(heads) for r in chosen[i]],
                                  device='cuda', dtype=torch.int32)
            dest = torch.arange(len(group) * heads * width, device='cuda', dtype=torch.int32)
            output = torch.empty((2, len(group) * heads * width, 1, dim),
                                 device='cuda', dtype=initial.dtype)
            h2d.kv_cache_h2d_transfer(pool, output, source, dest)
            outputs.append(output.reshape(2, len(group), heads, width, dim))
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start))
        actual = torch.cat(outputs, dim=1).cpu()
        expected_layers = []
        for i in range(layers):
            history = torch.cat([initial[i]] + [item[i].cpu() for item in incoming[:step + 1]], dim=2)
            expected_layers.append(history.index_select(2, torch.tensor(chosen[i], dtype=torch.long)))
        expected = torch.stack(expected_layers, dim=1)
        assert torch.equal(actual, expected), f'{kind} step {step}: read mismatch'
        reads.append(actual)
    expected_final = torch.cat([initial.permute(1, 0, 2, 3, 4)] +
                               [item.cpu().permute(1, 0, 2, 3, 4) for item in incoming], dim=3)
    assert torch.equal(view[:, :, :, :rows + len(incoming)], expected_final), 'final K/V mismatch'
    return samples, reads, pool.numel() * pool.element_size()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--build-dir', type=Path, default=Path('/data/nishant/tieredkv_external/flexicache_transfer_build_20261001'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = dict(passed=False, scope=__doc__, runtime=runtime_identity(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[])
    try:
        d2h, dhash = load_kernel(args.build_dir, 'd2h')
        h2d, hhash = load_kernel(args.build_dir, 'h2d')
        expected = json.loads(Path('results/baseline_fidelity/flexicache_transfer_check_v2_20261001.json').read_text())['binary_sha256']
        assert dict(d2h=dhash, h2d=hhash) == expected
        report.update(binary_sha256=expected, gpu=torch.cuda.get_device_name(), seed=0)
        generator = torch.Generator().manual_seed(0)
        for label, layers, rows, steps, capacity in [('empty', 2, 64, 1, 65), ('grow', 32, 1536, 20, 2081)]:
            initial = torch.randn((layers, 2, 8, rows, 128), generator=generator, dtype=torch.bfloat16)
            incoming = [torch.randn((layers, 2, 8, 1, 128), generator=generator, dtype=torch.bfloat16).cuda() for _ in range(steps)]
            selections = [[[] if label == 'empty' else [(step * 41 + layer * 13 + i * 47) % (rows + step + 1) for i in range(32)]
                           for layer in range(layers)] for step in range(steps)]
            for block in range(1 if label == 'empty' else 3):
                for kind in ('per_layer', 'batched', 'batched', 'per_layer'):
                    samples, _, reserved = arm(kind, initial, incoming, selections, capacity, d2h, h2d)
                    warmup = 4 if label == 'grow' else 0
                    report['rows'].append(dict(regime=label, block=block, arm=kind, warmup_ms=samples[:warmup], step_ms=samples[warmup:], reserved_host_bytes=reserved))
                    print(label, block, kind, round(statistics.median(samples), 3), flush=True)
                    gc.collect()
        summary = {}
        for kind in ('per_layer', 'batched'):
            samples = sorted(t for row in report['rows'] if row['regime'] == 'grow' and row['arm'] == kind for t in row['step_ms'])
            summary[kind] = dict(median_ms=statistics.median(samples), p95_ms=samples[int(.95 * (len(samples) - 1))])
        blocks = []
        for block in range(3):
            medians = {kind: statistics.median(t for row in report['rows'] if row['regime'] == 'grow' and row['block'] == block and row['arm'] == kind for t in row['step_ms']) for kind in summary}
            blocks.append(medians)
        saving = summary['per_layer']['median_ms'] - summary['batched']['median_ms']
        report.update(passed=True, exact_reads_and_final_kv=True, summary=summary, blocks=blocks, median_saving_ms=saving,
                      advance_to_model_trial=saving >= 4 and all(b['batched'] < b['per_layer'] for b in blocks) and summary['batched']['p95_ms'] <= 1.02 * summary['per_layer']['p95_ms'],
                      limitations='Optimistic storage-only screen; all layer inputs and indices available together. No schedulable model integration, whole-loop, overlap, quality or physical STT claim.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
