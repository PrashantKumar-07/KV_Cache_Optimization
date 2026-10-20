"""Charged exact CPU protected-mask construction falsification screen."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from tiered_kv_cache import TieredKVCache
from provenance import atomic_json, runtime_identity


def fast_mask(cache):
    if not cache.sram_pos:
        return torch.zeros(0,dtype=torch.bool)
    positions=np.asarray(cache.sram_pos)
    limit=np.iinfo(np.int64).max
    if positions.dtype != np.int64 or not 0 <= cache.cfg.sink_size <= limit or not 0 <= cache.cfg.window_size <= limit:
        return TieredKVCache._hard_protected_mask(cache)
    with np.errstate(over='ignore'):
        cutoff=positions.max()-np.int64(cache.cfg.window_size)
    return torch.from_numpy((positions<cache.cfg.sink_size)|(positions>cutoff))


def fixture(positions,sink=4,window=128):
    return SimpleNamespace(sram_pos=positions,cfg=SimpleNamespace(sink_size=sink,window_size=window))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=dict(passed=False,runtime=runtime_identity(),rows=[],checks=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,('experiments/probe_protected_mask_numpy.py','src/tiered_kv_cache.py'))})
    try:
        cases=[[],[0],list(range(31)),list(range(1024)),list(range(1057)),list(range(1057))[::-1],[-10,-3,0,1,100], [0,0,200,200], [-(1<<63),-(1<<63)+1],[(1<<63)-2,(1<<63)-1],[False,True],[0.,2.,140.]]
        for index,positions in enumerate(cases):
            for sink,window in ((4,128),(0,0),(4,4096)):
                cache=fixture(positions,sink,window)
                a=TieredKVCache._hard_protected_mask(cache);b=fast_mask(cache)
                assert a.dtype==b.dtype==torch.bool and torch.equal(a,b),(index,sink,window)
                report['checks'].append(dict(case=index,sink=sink,window=window,exact=True))
        caches=[fixture(list(range(4))+list(range(2000+layer*31,2000+layer*31+1053))) for layer in range(32)]
        for cache in caches:assert torch.equal(TieredKVCache._hard_protected_mask(cache),fast_mask(cache))
        for block in range(3):
            for arm in ('baseline','numpy','numpy','baseline'):
                samples=[]
                for step in range(20):
                    torch.cuda.synchronize();start=time.perf_counter()
                    values=[(TieredKVCache._hard_protected_mask if arm=='baseline' else fast_mask)(cache) for cache in caches]
                    torch.cuda.synchronize();samples.append(1000*(time.perf_counter()-start))
                report['rows'].append(dict(block=block,arm=arm,warmup_ms=samples[:4],step_ms=samples[4:]))
                print(block,arm,round(statistics.median(samples[4:]),3),flush=True)
        summary={arm:dict(median_ms=statistics.median(v for r in report['rows'] if r['arm']==arm for v in r['step_ms']),p95_ms=float(np.percentile([v for r in report['rows'] if r['arm']==arm for v in r['step_ms']],95))) for arm in ('baseline','numpy')}
        blocks=[{arm:statistics.median(v for r in report['rows'] if r['block']==block and r['arm']==arm for v in r['step_ms']) for arm in summary} for block in range(3)]
        saving=summary['baseline']['median_ms']-summary['numpy']['median_ms']
        report.update(passed=True,summary=summary,blocks=blocks,median_saving_ms=saving,advance_to_model=saving>=4 and all(b['numpy']<b['baseline'] for b in blocks) and summary['numpy']['p95_ms']<=1.02*summary['baseline']['p95_ms'],limitations='Synthetic32layerCPUmask construction only; exact predicates but no model/quality/wholeloop timing claim. CPUNumPy dtype guards/conversion/max/comparison charged. Existing cache code unchanged.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('summary','median_saving_ms','advance_to_model')},indent=2))


if __name__=='__main__':main()
