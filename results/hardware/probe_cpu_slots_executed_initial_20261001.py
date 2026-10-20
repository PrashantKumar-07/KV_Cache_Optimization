"""Storage-only CPU-fast versus pinned native slot-transfer screen."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from provenance import atomic_json, runtime_identity


def load_kernel(directory, name):
    path = directory / name / f'flexicache_check_{name}.so'
    spec = importlib.util.spec_from_file_location(f'flexicache_check_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def run_arm(kind, initial, incoming, selections, drops, capacity, d2h, h2d):
    layers, _, heads, rows, dim = initial.shape
    logical = list(range(rows))
    free = list(range(rows, capacity))
    states = [initial[layer].clone() for layer in range(layers)] if kind == 'packed' else None
    pool = None
    if kind == 'slots':
        pool = torch.zeros((2, layers * heads * capacity, 1, dim), dtype=initial.dtype, pin_memory=True)
        for layer in range(layers):
            view = pool[:, layer * heads * capacity:(layer + 1) * heads * capacity].reshape(2, heads, capacity, dim)
            view[:, :, :rows].copy_(initial[layer])
        h2d.register_cpu_kv_cache_pinned(pool)  # native module has one global mapped pointer
    samples, snapshots = [], []
    torch.cuda.synchronize()
    for step, (selected, dead) in enumerate(zip(selections, drops)):
        start = time.perf_counter()
        keep = [i for i in range(len(logical)) if i not in set(dead)]
        count = incoming[step].shape[3]
        if kind == 'slots':
            free.extend(logical[i] for i in dead)
            if len(free) < count:
                raise ValueError('insufficient transient slot capacity')
            appended, free = free[:count], free[count:]
            logical = [logical[i] for i in keep] + appended
        else:
            logical = list(range(len(keep) + count))
        reads = []
        for layer in range(layers):
            new = incoming[step][layer]
            if kind == 'packed':
                old = states[layer].index_select(2, torch.tensor(keep, dtype=torch.long)) if dead else states[layer]
                states[layer] = torch.cat((old, new.cpu()), dim=2)
                reads.append(states[layer].index_select(2, torch.tensor(selected, dtype=torch.long)).cuda())
            else:
                base = layer * heads * capacity
                dst = torch.tensor([base + head * capacity + slot for head in range(heads) for slot in appended], device='cuda', dtype=torch.int32)
                src = torch.arange(heads * count, device='cuda', dtype=torch.int32)
                d2h.kv_cache_d2h_transfer(new.reshape(2, heads * count, 1, dim), pool, src, dst)
                chosen = torch.tensor([base + head * capacity + logical[i] for head in range(heads) for i in selected], device='cuda', dtype=torch.int32)
                dest = torch.arange(heads * len(selected), device='cuda', dtype=torch.int32)
                read = torch.empty((2, heads * len(selected), 1, dim), device='cuda', dtype=initial.dtype)
                h2d.kv_cache_h2d_transfer(pool, read, chosen, dest)
                reads.append(read.reshape(2, heads, len(selected), dim))
        torch.cuda.synchronize()  # pinned pool remains owned until all host writes finish
        samples.append(1000 * (time.perf_counter() - start))
        snapshots.append([read.cpu() for read in reads])
    if kind == 'packed':
        final = states
    else:
        final = [pool[:, layer * heads * capacity:(layer + 1) * heads * capacity].reshape(2, heads, capacity, dim).index_select(2, torch.tensor(logical)) for layer in range(layers)]
    reserved = (pool.numel() * pool.element_size()) if pool is not None else sum(x.numel() * x.element_size() for x in states)
    return samples, snapshots, final, reserved


def same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=Path('/data/nishant/tieredkv_external/flexicache_transfer_build_20261001'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = {'scope': __doc__, 'runtime': runtime_identity(), 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'passed': False, 'rows': []}
    try:
        d2h, dhash = load_kernel(args.build_dir, 'd2h')
        h2d, hhash = load_kernel(args.build_dir, 'h2d')
        frozen = json.loads(Path('results/baseline_fidelity/flexicache_transfer_check_v2_20261001.json').read_text())
        assert {'d2h': dhash, 'h2d': hhash} == frozen['binary_sha256'], 'kernel binaries differ from verified source build'
        report.update(binary_sha256={'d2h': dhash, 'h2d': hhash}, gpu=torch.cuda.get_device_name())
        generator = torch.Generator().manual_seed(0)
        # Separate recycle and zero-read checks; measurements use actual short-decode fill.
        for label, layers, rows, capacity, steps, count in [('recycle', 2, 64, 64, 4, 3), ('grow', 32, 1536, 2081, 16, 1)]:
            initial = torch.randn((layers, 2, 8, rows, 128), generator=generator, dtype=torch.bfloat16)
            incoming = [torch.randn((layers, 2, 8, count, 128), generator=generator, dtype=torch.bfloat16).cuda() for _ in range(steps)]
            drops = [[(step * 7 + i) % rows for i in range(count)] if label == 'recycle' else [] for step in range(steps)]
            selections = [([] if label == 'recycle' and step == 0 else [(step * 41 + i * 47) % (rows if label == 'recycle' else rows + step + 1) for i in range(32)]) for step in range(steps)]
            reference = None
            for block in range(1 if label == 'recycle' else 3):
                for kind in ('packed', 'slots', 'slots', 'packed'):
                    samples, reads, final, reserved = run_arm(kind, initial, incoming, selections, drops, capacity, d2h, h2d)
                    if reference is None:
                        reference = (reads, final)
                    else:
                        assert all(same(a, b) for a, b in zip(reads, reference[0])) and same(final, reference[1]), f'{label}/{kind}: exact K/V mismatch'
                    report['rows'].append(dict(regime=label, block=block, arm=kind, step_ms=samples, reserved_host_bytes=reserved, final_occupied_host_bytes=layers * 2 * 8 * (rows if label == 'recycle' else rows + steps) * 128 * 2))
                    print(label, block, kind, statistics.median(samples), flush=True)
                    gc.collect()
            del initial, incoming, reference, reads, final
        summary = {}
        for kind in ('packed', 'slots'):
            samples = sorted(t for row in report['rows'] if row['regime'] == 'grow' and row['arm'] == kind for t in row['step_ms'])
            summary[kind] = {'median_ms': statistics.median(samples), 'p95_ms': samples[int(.95 * (len(samples) - 1))]}
        report.update(summary=summary, exact_reads_and_final_kv=True, passed=True)
        report['advance_to_model_trial'] = summary['slots']['median_ms'] <= .95 * summary['packed']['median_ms'] and summary['slots']['p95_ms'] <= 1.02 * summary['packed']['p95_ms']
        report['limitations'] = 'Storage-only synchronous per-layer transfers; no model, policy, prefill, physical STT-RAM or overlap claim. Pool ownership extends through synchronization; one registered native source pool at a time.'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
