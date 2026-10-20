"""Charged complete victim-selection stage, exact production AST reference."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import textwrap
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from longbench_eval import TieredKVCache
from provenance import atomic_json, runtime_identity
from probe_protected_mask_numpy import fast_mask


def reference_function():
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVCache.evict_and_demote)))
    node=tree.body[0]
    first=next(i for i,x in enumerate(node.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='n_over' for t in x.targets))
    last=next(i for i,x in enumerate(node.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='backup_by_pos' for t in x.targets))
    body=[ast.Assign(targets=[ast.Name(id='cfg',ctx=ast.Store())],value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='cfg',ctx=ast.Load()))]+copy.deepcopy(node.body[first:last])+[ast.Return(value=ast.Name(id='victims',ctx=ast.Load()))]
    f=ast.FunctionDef(name='reference',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]));env={'torch':torch}
    exec(compile(module,'<production_victim_selection>','exec'),env)
    return env['reference'],ast.dump(module,include_attributes=False)


def batched_victims(caches, numpy_masks=False):
    requests=[max(0,len(c.sram_pos)-c.cfg.sram_capacity) for c in caches]
    active=[i for i,n in enumerate(requests) if n]
    result=[[] for _ in caches]
    if not active:return result
    chosen=[caches[i] for i in active]
    assert all(c.cfg.write_aware_lambda==0 for c in chosen)
    hard=[fast_mask(c) if numpy_masks else c._hard_protected_mask() for c in chosen]
    soft=[h|c.sram_grace for h,c in zip(hard,chosen)]
    masks=torch.stack((pad_sequence(hard,batch_first=True,padding_value=True),pad_sequence(soft,batch_first=True,padding_value=True))).to(chosen[0].sram_cum.device)
    cumulative=pad_sequence([c.sram_cum for c in chosen],batch_first=True)
    age=pad_sequence([c.sram_age for c in chosen],batch_first=True)
    score=cumulative/age.clamp(min=1)
    soft_score=score.masked_fill(masks[1],float('inf'))
    order=torch.argsort(soft_score,dim=1,stable=True)
    finite=torch.isfinite(soft_score).sum(dim=1)
    width=min(max(requests[i] for i in active),order.shape[1])
    returned=torch.cat((finite[:,None],order[:,:width]),dim=1).tolist()
    fallback=[]
    for row,(index,values) in enumerate(zip(active,returned)):
        take=min(requests[index],values[0])
        result[index]=values[1:1+take]
        if take<requests[index]:fallback.append(row)
    if fallback:
        hard_score=score[fallback].clone()
        already=torch.zeros_like(hard_score,dtype=torch.bool)
        for row,original in enumerate(fallback):
            already[row,result[active[original]]]=True
        hard_score.masked_fill_(masks[0,fallback]|already,float('inf'))
        hard_order=torch.argsort(hard_score,dim=1,stable=True)
        finite=torch.isfinite(hard_score).sum(dim=1)
        returned=torch.cat((finite[:,None],hard_order[:,:width]),dim=1).tolist()
        for original,values in zip(fallback,returned):
            index=active[original];take=min(requests[index]-len(result[index]),values[0])
            result[index]+=values[1:1+take]
    return result


def cache(positions,cumulative,age,capacity=1024,grace=None,sink=4,window=128):
    return SimpleNamespace(sram_pos=positions,sram_cum=cumulative,sram_age=age,
        sram_grace=torch.zeros(len(positions),dtype=torch.bool) if grace is None else grace,
        cfg=SimpleNamespace(sram_capacity=capacity,sink_size=sink,window_size=window,write_aware_lambda=0.,inclusive=True),
        _hard_protected_mask=lambda:TieredKVCache._hard_protected_mask(SimpleNamespace(sram_pos=positions,cfg=SimpleNamespace(sink_size=sink,window_size=window))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--numpy-masks',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    reference,source=reference_function()
    report=dict(passed=False,numpy_masks=args.numpy_masks,runtime=runtime_identity(),reference_ast_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=[],checks=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,('experiments/probe_batched_eviction_selection.py','experiments/probe_protected_mask_numpy.py','src/tiered_kv_cache.py'))})
    try:
        torch.manual_seed(0)
        primary=[]
        for layer in range(32):
            positions=list(range(4))+list(range(1000+layer*13,1892+layer*13))+list(range(16256,16384))+list(range(5000+layer*33,5032+layer*33))+[16384]
            grace=torch.zeros(1057,dtype=torch.bool);grace[1024:1056]=True
            cumulative=torch.randint(0,64,(1057,),device='cuda').float()/64
            age=torch.randint(0,64,(1057,),device='cuda').float()
            primary.append(cache(positions,cumulative,age,grace=grace))
        assert batched_victims(primary,args.numpy_masks)==[reference(c) for c in primary]
        report['checks'].append(dict(case='primary32',exact=True))
        for label in ('empty_nooverflow','ragged_ties','grace_fallback','all_protected','nonfinite','subnormal'):
            cases=[]
            for layer,size in enumerate((0,1,33,64,97,1057)):
                positions=list(range(size));capacity=max(0,size-17)
                cumulative=torch.ones(size,device='cuda');age=torch.ones(size,device='cuda')
                grace=torch.zeros(size,dtype=torch.bool)
                if label=='empty_nooverflow':capacity=size
                if label=='grace_fallback':grace[:]=True
                if label=='all_protected':window=4096
                else:window=8
                if label=='nonfinite' and size>12:
                    cumulative[4:8]=torch.tensor([float('nan'),float('inf'),-float('inf'),0.],device='cuda')
                if label=='subnormal' and size:
                    cumulative[:]=torch.nextafter(torch.tensor(0.,device='cuda'),torch.tensor(1.,device='cuda'))
                cases.append(cache(positions,cumulative,age,capacity,grace,window=window if label!='empty_nooverflow' else 8))
            assert batched_victims(cases,args.numpy_masks)==[reference(c) for c in cases],label
            report['checks'].append(dict(case=label,exact=True))
        for block in range(3):
            for arm in ('baseline','batched','batched','baseline'):
                values=[]
                for step in range(20):
                    torch.cuda.synchronize();start=time.perf_counter()
                    actual=[reference(c) for c in primary] if arm=='baseline' else batched_victims(primary,args.numpy_masks)
                    torch.cuda.synchronize();values.append(1000*(time.perf_counter()-start))
                report['rows'].append(dict(block=block,arm=arm,warmup_ms=values[:4],step_ms=values[4:]))
                print(block,arm,round(statistics.median(values[4:]),3),flush=True)
        summary={a:dict(median_ms=statistics.median(v for r in report['rows'] if r['arm']==a for v in r['step_ms']),p95_ms=float(np.percentile([v for r in report['rows'] if r['arm']==a for v in r['step_ms']],95))) for a in ('baseline','batched')}
        blocks=[{a:statistics.median(v for r in report['rows'] if r['block']==b and r['arm']==a for v in r['step_ms']) for a in summary} for b in range(3)]
        saving=summary['baseline']['median_ms']-summary['batched']['median_ms']
        report.update(passed=True,summary=summary,blocks=blocks,median_saving_ms=saving,advance_to_model=saving>=4 and all(b['batched']<b['baseline'] for b in blocks) and summary['batched']['p95_ms']<=1.02*summary['baseline']['p95_ms'],limitations='Synthetic complete selection stage afterpromotion/append, excludes preparing those states. CPUmask construction/padding/GPUcopies/arithmetic/stableties/finitecounts/return/fallback charged. lambda0 only; no cache edits/model/quality/fullloop speed claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('summary','median_saving_ms','advance_to_model')},indent=2))


if __name__=='__main__':main()
