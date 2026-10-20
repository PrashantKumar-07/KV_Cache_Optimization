"""Charged projected-metadata/victim-selection stage before model integration."""
import argparse
import hashlib
from pathlib import Path
import statistics
from types import SimpleNamespace
import time
import numpy as np
import torch
import projected_metadata_batch as candidate
import probe_batched_eviction_selection as original
from provenance import atomic_json, runtime_identity


def rows(sizes, *, mode='normal'):
    result = []
    for layer, size in enumerate(sizes):
        positions = list(range(size))
        if size == 1056:
            positions = list(range(4)) + list(range(1000+layer*13, 1892+layer*13)) + list(range(16256, 16384)) + list(range(5000+layer*33, 5032+layer*33))
        cumulative = torch.randint(0, 64, (size,), device='cuda').float()/64
        age = torch.randint(0, 64, (size,), device='cuda').float()
        grace = torch.zeros(size, dtype=torch.bool)
        if size == 1056: grace[-32:] = True
        if mode == 'grace': grace[:] = True
        if mode == 'nonfinite' and size >= 4:
            cumulative[:4] = torch.tensor([float('nan'), float('inf'), -float('inf'), -0.], device='cuda')
        if mode == 'large_age': age[:] = torch.tensor(2**24, device='cuda', dtype=torch.float32)
        if mode == 'subnormal': cumulative[:] = torch.nextafter(torch.tensor(0., device='cuda'), torch.tensor(1., device='cuda'))
        cache = original.cache(positions, cumulative, age, 1024 if size == 1056 else max(0, size-17), grace=grace, window=128 if size==1056 else 8)
        cache.cfg.device = 'cuda'; cache.cfg.dtype = torch.float64 if mode == 'mixed_dtype' and layer%2 else torch.float32
        result.append((cache, positions+[16384]))
    return result


def project(values, n_new, fast):
    prepared = candidate.packed_projection(values, n_new) if fast else candidate.reference_projection(values, n_new)
    selected = (candidate.STACKED_VICTIMS if fast else original.batched_victims)(prepared, numpy_masks=True)
    return prepared, selected


def same_tensor(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    files = list(Path('src').glob('*.py')) + list(Path('experiments').glob('*.py'))
    report = dict(passed=False, checks=[], rows=[], runtime=runtime_identity(), ast=candidate.source_identity(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    try:
        torch.manual_seed(0)
        fixtures = [('empty', []), ('single_empty', [0]), ('uniform32', [1056]*32),
            ('ragged', [0,1,33,64,97,1056]), ('grace', [1056]*4), ('nonfinite', [1056]*4),
            ('large_age', [1056]*4), ('subnormal', [1056]*4), ('mixed_dtype', [64]*4)]
        for label, sizes in fixtures:
            for n_new in (1, 32):
                values = rows(sizes, mode=label)
                # New positions must agree with the exact requested append count.
                values = [(cache, cache.sram_pos+list(range(16384, 16384+n_new))) for cache, _ in values]
                before = [(c.sram_cum.clone(), c.sram_age.clone(), c.sram_grace.clone(), list(c.sram_pos)) for c, _ in values]
                a, indices_a = project(values, n_new, False); b, indices_b = project(values, n_new, True)
                assert indices_a == indices_b, (label, n_new)
                for ref, test in zip(a, b):
                    assert ref.sram_pos == test.sram_pos
                    for field in ('sram_cum', 'sram_age', 'sram_grace'): assert same_tensor(getattr(ref, field), getattr(test, field)), (label, n_new, field)
                for (cache, _), prior in zip(values, before):
                    assert same_tensor(cache.sram_cum, prior[0]) and same_tensor(cache.sram_age, prior[1]) and same_tensor(cache.sram_grace, prior[2]) and cache.sram_pos == prior[3]
                report['checks'].append(dict(case=label, n_new=n_new, exact=True, inputs_unmutated=True))
        primary = rows([1056]*32)
        reference = project(primary, 1, False)[1]
        for block in range(3):
            for arm in ('control', 'batched', 'batched', 'control'):
                values = []
                for step in range(20):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    _, actual = project(primary, 1, arm=='batched')
                    torch.cuda.synchronize(); values.append(1000*(time.perf_counter()-start))
                    assert actual == reference
                report['rows'].append(dict(block=block, arm=arm, warmup_ms=values[:4], step_ms=values[4:]))
                atomic_json(args.output, report)
                print(block, arm, round(statistics.median(values[4:]), 3), flush=True)
        summary = {arm:dict(median_ms=statistics.median(v for row in report['rows'] if row['arm']==arm for v in row['step_ms']),
                        p95_ms=float(np.percentile([v for row in report['rows'] if row['arm']==arm for v in row['step_ms']],95))) for arm in ('control','batched')}
        blocks = [{arm:statistics.median(v for row in report['rows'] if row['block']==block and row['arm']==arm for v in row['step_ms']) for arm in summary} for block in range(3)]
        saved = summary['control']['median_ms'] - summary['batched']['median_ms']
        report.update(passed=True, summary=summary, blocks=blocks, median_saving_ms=saved,
            advance_to_model=saved>=4 and all(b['batched']<b['control'] for b in blocks) and summary['batched']['p95_ms']<=summary['control']['p95_ms']*1.02,
            limitations='Synthetic post-promotion metadata, one seed/geometry. All projection allocation/packing, CPU masks, ranking/fallback, return and sync charged. Actual promotion/cold transfer, model, quality and full-loop memory/speed excluded; stage only.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
