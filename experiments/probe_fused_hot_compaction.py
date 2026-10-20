"""Charged complete hot-compaction screen against the production AST."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from fused_hot_compaction import compact_group,kernel
from probe_shared_hot_index import equal,fixture,reference,run
from provenance import atomic_json,runtime_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    control,body=reference()
    sources=list(Path('src').glob('*.py'))+list(Path('experiments').glob('*.py'))+[
        Path('experiments/native_hot_compact.cpp'),Path('experiments/native_hot_compact.cu')]
    report=dict(passed=False,runtime=runtime_identity(),rows=[],checks=[],
        reference_ast_sha256=hashlib.sha256(body.encode()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    try:
        module=kernel()
        report['binary_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        report['gpu']=torch.cuda.get_device_name()
        torch.manual_seed(0)
        for size in (0,1,32,1057):
            states=[fixture(size) for _ in range(3)]
            for state in states:
                if size>2:
                    state.sram_cum[:3]=torch.tensor([float('nan'),float('inf'),-float('inf')],device='cuda')
                    state.sram_k[:,:3,:]=torch.tensor([float('nan'),float('inf'),-float('inf')],device='cuda',dtype=torch.bfloat16)[None,:,None]
            for label,selection in (('none',[]),('all',list(range(size))),('onekept',list(range(max(0,size-1))))):
                victims=[selection[::-1] for _ in states]
                equal(run(control,states,victims),compact_group(states,victims))
                report['checks'].append(dict(rows=size,selection=label,exact=True))
        states=[fixture(1057+layer%3) for layer in range(32)]
        victims=[list(range(layer,layer+len(state.sram_pos)-1024))[::-1] for layer,state in enumerate(states)]
        equal(run(control,states,victims),compact_group(states,victims))
        report['checks'].append(dict(case='ragged_sources_same_kept_width',exact=True))
        for block in range(3):
            for arm in ('control','fused','fused','control'):
                values=[];outputs=None
                for step in range(20):
                    outputs=None
                    torch.cuda.synchronize();start=time.perf_counter()
                    outputs=run(control,states,victims) if arm=='control' else compact_group(states,victims)
                    torch.cuda.synchronize();values.append(1000*(time.perf_counter()-start))
                equal(outputs,run(control,states,victims))
                report['rows'].append(dict(block=block,arm=arm,warmup_ms=values[:4],step_ms=values[4:]))
                atomic_json(args.output,report)
                print(block,arm,statistics.median(values[4:]),flush=True)
        summary={arm:dict(median_ms=statistics.median(v for r in report['rows'] if r['arm']==arm for v in r['step_ms']),
            p95_ms=float(np.percentile([v for r in report['rows'] if r['arm']==arm for v in r['step_ms']],95))) for arm in ('control','fused')}
        blocks=[{arm:statistics.median(v for r in report['rows'] if r['block']==block and r['arm']==arm for v in r['step_ms']) for arm in summary} for block in range(3)]
        saving=summary['control']['median_ms']-summary['fused']['median_ms']
        report.update(passed=True,summary=summary,blocks=blocks,median_saving_ms=saving,
            advance_to_model=saving>=4 and all(b['fused']<b['control'] for b in blocks) and summary['fused']['p95_ms']<=1.02*summary['control']['p95_ms'],
            limitations='Complete post-victim hot compaction only. CPUmask/index/position/grace, descriptors/uploads, recordstream, validation, allocation and allKV/F32score gathers charged. One homogeneous BF16/F32 GPU and same kept width; binaryJIT build/warmup outside timing. No full-model/quality/default claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('summary','median_saving_ms','advance_to_model')},indent=2))


if __name__=='__main__':main()
