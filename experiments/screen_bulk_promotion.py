"""Exact replay and charged CPU promotion-planning screen on captured inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import bulk_promotion_stage as stage
from provenance import atomic_json, runtime_identity


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def check(frame, captured=False):
    outputs, states = [], []
    for fast in (False, True):
        policy, caches = stage.fixture(frame, fast)
        outputs.append(stage.run(policy, caches, fast))
        states.append(caches)
        for layer, cache in caches.items():
            for n in ('stt_shadow','stt_was_vram_resident'):
                assert same(getattr(cache, n), frame['layers'][layer][n]), n
            assert cache.stt_pos == frame['layers'][layer]['positions']
            if captured: assert same(cache.stt_peek_streak, frame['expected_streaks'][layer])
        if captured:
            assert outputs[-1][0] == frame['expected_plans']
            assert outputs[-1][1] == frame['expected_selections']
    assert outputs[0] == outputs[1]
    for layer in states[0]: assert same(states[0][layer].stt_peek_streak, states[1][layer].stt_peek_streak)


def synthetic(mode, inclusive, floor, dtype=torch.float32):
    size = 0 if mode == 'empty' else 9
    shadow = torch.tensor([i % 3 == 0 or mode == 'all_shadow' for i in range(size)], dtype=torch.bool)
    resident = torch.tensor([i % 2 == 0 for i in range(size)], dtype=torch.bool)
    streak = torch.arange(size, dtype=dtype)
    if mode == 'nonfinite': streak[:3] = torch.tensor([float('nan'),float('inf'),-float('inf')],dtype=dtype)
    if mode == 'large_streak': streak[:] = 2**24
    if mode == 'subnormal': streak[:] = torch.nextafter(torch.tensor(0.,dtype=dtype),torch.tensor(1.,dtype=dtype))
    positions = list(range(100,100+size))
    attention = {p: 1. for p in positions}
    indexed = {i: 1. for i in range(size)}
    if mode == 'nonfinite':
        indexed.update({1:float('nan'),2:float('inf'),3:-float('inf')})
    if mode == 'position_only': indexed = {}
    if mode == 'no_attention': attention = {}
    return dict(budget=5, attention={0:attention}, attention_indices={0:indexed},
        layers={0:dict(positions=positions, inclusive=inclusive, floor=floor, hysteresis=2,
            stt_shadow=shadow,stt_was_vram_resident=resident,stt_peek_streak=streak)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args(); target = args.directory/'stage.json'
    if target.exists(): raise FileExistsError(target)
    capture = json.loads((args.directory/'capture_report.json').read_text())
    inputs = args.directory/'inputs.pt'
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == capture['input_sha256']
    frames = torch.load(inputs,weights_only=False)
    files = ('experiments/bulk_promotion_stage.py','experiments/screen_bulk_promotion.py',
        'experiments/longbench_eval.py','src/tiered_kv_cache.py')
    report = dict(passed=False, runtime=runtime_identity(), threads=torch.get_num_threads(),
        input_sha256=capture['input_sha256'], source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        ast=dict(plan=stage.PLAN_AST,rows=stage.ROWS_AST),checks=[],rows=[])
    try:
        for frame in frames: check(frame,captured=True)
        report['captured_layer_checks'] = sum(len(f['layers']) for f in frames)
        modes=('ties','nonfinite','subnormal','large_streak','all_shadow','position_only','no_attention','empty')
        for mode in modes:
            for inclusive in (False,True):
                for floor in (0,1,9):
                    frame = synthetic(mode,inclusive,floor); check(frame)
                    # Direct resolver inputs cover invalid indices and stable de-duplication.
                    results, states = [], []
                    for fast in (False,True):
                        _, caches = stage.fixture(frame,fast); cache = caches[0]
                        results.append(cache.resolve_promotion_indices([-1,99,1,1,2,3,8]))
                        states.append(cache.stt_peek_streak)
                    assert results[0] == results[1] and same(*states)
                    report['checks'].append(dict(mode=mode,inclusive=inclusive,floor=floor,exact=True))
        for dtype in (torch.float64,torch.bfloat16):
            check(synthetic('ties',True,1,dtype))
            report['checks'].append(dict(mode=str(dtype),exact=True,fallback=True))
        for block in range(3):
            for arm in ('control','bulk','bulk','control'):
                values=[]
                for step in range(30):
                    frame=frames[step%len(frames)]
                    policy,caches=stage.fixture(frame,arm=='bulk')
                    start=time.perf_counter()
                    plans,selections,_=stage.run(policy,caches,arm=='bulk')
                    elapsed=1000*(time.perf_counter()-start)
                    assert plans == frame['expected_plans'] and selections == frame['expected_selections']
                    for layer,cache in caches.items(): assert same(cache.stt_peek_streak,frame['expected_streaks'][layer])
                    values.append(elapsed)
                report['rows'].append(dict(block=block,arm=arm,warmup_ms=values[:10],step_ms=values[10:]))
                atomic_json(target,report)
                print(block,arm,round(statistics.median(values[10:]),3),flush=True)
        summary={arm:dict(median_ms=statistics.median(v for r in report['rows'] if r['arm']==arm for v in r['step_ms']),
            p95_ms=float(np.percentile([v for r in report['rows'] if r['arm']==arm for v in r['step_ms']],95))) for arm in ('control','bulk')}
        blocks=[{arm:statistics.median(v for r in report['rows'] if r['block']==b and r['arm']==arm for v in r['step_ms']) for arm in summary} for b in range(3)]
        saving=summary['control']['median_ms']-summary['bulk']['median_ms']
        report.update(passed=True,summary=summary,blocks=blocks,median_saving_ms=saving,
            advance_to_model=saving>=4 and all(b['bulk']<b['control'] for b in blocks) and summary['bulk']['p95_ms']<=summary['control']['p95_ms']*1.02,
            limitations='Actual fixed-development promotion inputs, one seed/geometry, CPU metadata stage only. Ranking, hysteresis mutation, bulk-view/array conversion, both row/floor checks charged. Fixture cloning/reset and result checks excluded equally. KV movement, transfer/native dispatch, model, quality and whole-loop resource/speed excluded.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'; raise
    finally: atomic_json(target,report)


if __name__ == '__main__': main()
