"""Exact-bit and charged compaction/visible-output screen for raw-buffer reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import numpy as np
import torch
from fused_hot_compaction import gather,kernel
from provenance import atomic_json,runtime_identity
from reuse_visible_policy import can_reuse,gather_into,ReuseVisiblePolicy,CONTROL_AST
GATHER_INTO=gather_into


def fixture(layers,hot,quota,*,bits=False):
    rows=64 if bits else hot+quota+1
    dtype=torch.bfloat16;states=[];raw=[];cold=[]
    for _ in range(layers):
        keys=torch.arange(8*rows*128,device='cuda',dtype=torch.int32).to(torch.int16).view(dtype).reshape(8,rows,128)
        metadata=torch.tensor([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc00001,0x7f800001],
            device='cuda',dtype=torch.int64).to(torch.int32).view(torch.float32).repeat((rows+7)//8)[:rows]
        states.append(SimpleNamespace(sram_k=keys,sram_v=keys.flip(0),sram_cum=metadata,sram_age=metadata.flip(0),
            sram_pos=list(range(rows)),sram_grace=torch.arange(rows)%3==0))
        raw.append(tuple(torch.full((1,8,hot+quota+1,128),-777,dtype=dtype,device='cuda') for _ in range(2)))
        cold.append(tuple(torch.full((8,quota,128),12+channel,dtype=dtype,device='cuda') for channel in range(2)))
    return states,raw,cold


def stage(inputs,raw,cold,keeps,arm):
    states=[]
    for original,keep in zip(inputs,keeps):
        state=SimpleNamespace(**vars(original))
        state.pending_keep=np.asarray(keep,dtype=np.int32)
        state.sram_pos=np.asarray(original.sram_pos,dtype=np.int64)[state.pending_keep].tolist()
        state.sram_grace=torch.from_numpy(original.sram_grace.numpy()[state.pending_keep])
        states.append(state)
    if arm=='control':gather(states,keeps)
    else:
        assert can_reuse(states,raw,cold[0][0].shape[1])
        GATHER_INTO(states,raw)
    visible=[]
    for state,destination,selected in zip(states,raw,cold):
        visible.append(tuple((torch.cat((fast,tail),dim=1) if arm=='control' else
            ReuseVisiblePolicy._compose_visible(fast,tail,original)).unsqueeze(0)
            for fast,tail,original in zip((state.sram_k,state.sram_v),selected,destination)))
    return states,visible


def equal(a,b):
    for x,y in zip(a[0],b[0]):
        assert x.sram_pos==y.sram_pos and torch.equal(x.sram_grace,y.sram_grace)
        for name in ('sram_k','sram_v','sram_cum','sram_age'):
            assert torch.equal(getattr(x,name).contiguous().view(torch.uint8),getattr(y,name).contiguous().view(torch.uint8))
    for x,y in zip(a[1],b[1]):
        for first,second in zip(x,y):assert torch.equal(first.contiguous().view(torch.uint8),second.contiguous().view(torch.uint8))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    files=list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))+[
        Path('experiments/native_hot_compact.cpp'),Path('experiments/native_hot_compact.cu'),Path('experiments/native_reuse_hot.cu')]
    report=dict(passed=False,runtime=runtime_identity(),rows=[],checks=[],
        native_backend=args.native,
        original_visible_ast_sha256=hashlib.sha256(CONTROL_AST.encode()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    try:
        if args.native:
            global GATHER_INTO
            import native_reuse_visible
            GATHER_INTO=native_reuse_visible.gather_into
            report['reuse_binary_sha256']=hashlib.sha256(Path(native_reuse_visible.kernel().__file__).read_bytes()).hexdigest()
        report['binary_sha256']=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest()
        report['gpu']=torch.cuda.get_device_name();torch.manual_seed(0)
        for keep in (list(range(64)),list(range(0,64,2)),list(reversed(range(64))),[63,0,4,4]):
            states,raw,cold=fixture(4,len(keep),4,bits=True)
            snapshots=[t.clone() for pair in raw for t in pair]
            a=stage(states,raw,cold,[keep]*4,'control')
            b=stage(states,raw,cold,[keep]*4,'reuse');equal(a,b)
            for pair,old_pair in zip(raw,[snapshots[i:i+2] for i in range(0,len(snapshots),2)]):
                for new,old in zip(pair,old_pair):assert torch.equal(new[:,:,len(keep)+4:].view(torch.uint8),old[:,:,len(keep)+4:].view(torch.uint8))
            assert all(t.untyped_storage()._cdata==destination.untyped_storage()._cdata
                for pair,dest in zip(b[1],raw) for t,destination in zip(pair,dest))
            report['checks'].append(dict(kept_rows=len(keep),layers=4,exact_bits=True,tail_untouched=True,visible_aliases_raw=True))
        del states,raw,cold,snapshots,a,b
        for name,hot in (('shared',991),('default',1024)):
            states,raw,cold=fixture(32,hot,32);keeps=[list(range(hot))]*32
            for block in range(3):
                for arm in ('control','reuse','reuse','control'):
                    times=[];peaks=[];output=None
                    for trial in range(20):
                        output=None;torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        output=stage(states,raw,cold,keeps,arm)
                        torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
                        peaks.append(torch.cuda.max_memory_allocated()-baseline)
                    reference=stage(states,raw,cold,keeps,'control');equal(reference,output);del reference
                    report['rows'].append(dict(geometry=name,hot=hot,block=block,arm=arm,
                        warmup_ms=times[:4],step_ms=times[4:],peak_incremental_bytes=peaks[4:]))
                    atomic_json(args.output,report)
                    print(name,block,arm,statistics.median(times[4:]),flush=True)
            del states,raw,cold,output;torch.cuda.empty_cache()
        summaries={}
        for geometry in ('shared','default'):
            summary={}
            for arm in ('control','reuse'):
                rows=[r for r in report['rows'] if r['geometry']==geometry and r['arm']==arm]
                times=[v for row in rows for v in row['step_ms']]
                summary[arm]=dict(median_ms=statistics.median(times),p95_ms=float(np.percentile(times,95)),
                    maximum_incremental_gpu_bytes=max(v for r in rows for v in r['peak_incremental_bytes']))
            a,b=summary['control'],summary['reuse']
            summaries[geometry]=dict(summary=summary,median_saving_ms=a['median_ms']-b['median_ms'],
                charged_4ms_speed_screen_passed=a['median_ms']-b['median_ms']>=4,
                storage_candidate_regression_guard=b['median_ms']<=1.02*a['median_ms'] and b['p95_ms']<=1.02*a['p95_ms'],
                incremental_storage_reduced=b['maximum_incremental_gpu_bytes']<a['maximum_incremental_gpu_bytes'])
        report.update(passed=True,geometries=summaries,
            limitations='Charged CPUposition/grace preparation, guards/indexuploads, hot/metadata compaction, output allocation and visibleKVcomposition. Coldselection/transfers identical and excluded, fixtureimmutableinputs retained in both arms. Incremental GPUpeak excludes rawdestinationallocatedbeforetiming; whole-model/lifetimetrace needed. Reuse intentionally consumes decode buffers; no new kernel/default/CLI or speed/memory superiority claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)
    print(json.dumps(report['geometries'],indent=2))


if __name__=='__main__':main()
