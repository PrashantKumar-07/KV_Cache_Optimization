"""Charged full hot-state compaction stage; exact production AST control."""
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
from longbench_eval import TieredKVCache
from provenance import atomic_json, runtime_identity


FIELDS=('sram_k','sram_v','sram_pos','sram_cum','sram_age','sram_grace')


def reference():
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVCache.evict_and_demote)))
    node=next(n for n in tree.body[0].body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='victims')
    function=ast.FunctionDef(name='compact',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='victims')],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id='cfg',ctx=ast.Store())],value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='cfg',ctx=ast.Load())),copy.deepcopy(node)],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    env={'torch':torch};exec(compile(module,'<production_hot_compaction>','exec'),env)
    return env['compact'],ast.dump(module)


def shared_index(state,victims):
    if not victims:return
    mask=torch.ones(len(state.sram_pos),dtype=torch.bool);mask[victims]=False
    cpu_index=torch.where(mask)[0]
    keep=cpu_index.tolist()
    index=cpu_index.to(state.sram_k.device)
    state.sram_k=state.sram_k.index_select(1,index)
    state.sram_v=state.sram_v.index_select(1,index)
    state.sram_cum=state.sram_cum.index_select(0,index)
    state.sram_age=state.sram_age.index_select(0,index)
    state.sram_grace=state.sram_grace[mask]
    state.sram_pos=[state.sram_pos[i] for i in keep]


def fixture(rows):
    return SimpleNamespace(sram_k=torch.randn(8,rows,128,device='cuda',dtype=torch.bfloat16),
        sram_v=torch.randn(8,rows,128,device='cuda',dtype=torch.bfloat16),
        sram_pos=list(range(1000,1000+rows)),sram_cum=torch.randn(rows,device='cuda'),
        sram_age=torch.rand(rows,device='cuda'),sram_grace=torch.arange(rows)%2==0,
        cfg=SimpleNamespace(bool_hot_compaction=True))


def run(function,states,victims):
    outputs=[]
    for state,selection in zip(states,victims):
        output=SimpleNamespace(**vars(state))
        function(output,selection);outputs.append(output)
    return outputs


def equal(left,right):
    assert len(left)==len(right)
    for a,b in zip(left,right):
        for name in FIELDS:
            x,y=getattr(a,name),getattr(b,name)
            if isinstance(x,torch.Tensor):
                # Gather is bit-preserving, including nonfinite payloads.
                assert x.shape==y.shape and x.dtype==y.dtype
                assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)),name
            else:assert x==y,name


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    control,body=reference()
    report=dict(passed=False,runtime=runtime_identity(),rows=[],checks=[],reference_ast_sha256=hashlib.sha256(body.encode()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('src/tiered_kv_cache.py'))})
    try:
        torch.manual_seed(0)
        for size in (0,1,32,1057):
            state=fixture(size)
            if size>2:state.sram_cum[:3]=torch.tensor([float('nan'),float('inf'),-float('inf')],device='cuda')
            for label,selected in (('none',[]),('all',list(range(size))),('onekept',list(range(max(0,size-1))))):
                selected=selected[::-1]
                equal(run(control,[state],[selected]),run(shared_index,[state],[selected]))
                report['checks'].append(dict(rows=size,selection=label,exact=True))
        states=[fixture(1057) for _ in range(32)]
        victims=[list(range(layer,layer+33))[::-1] for layer in range(32)]
        equal(run(control,states,victims),run(shared_index,states,victims))
        for block in range(3):
            for arm in ('control','shared_index','shared_index','control'):
                function=control if arm=='control' else shared_index
                values=[];outputs=None
                for step in range(20):
                    # Previous outputs discarded before the charged new allocation.
                    outputs=None;torch.cuda.synchronize();start=time.perf_counter()
                    outputs=run(function,states,victims)
                    torch.cuda.synchronize();values.append(1000*(time.perf_counter()-start))
                equal(outputs,run(control,states,victims))
                report['rows'].append(dict(block=block,arm=arm,warmup_ms=values[:4],step_ms=values[4:]))
                atomic_json(args.output,report)
                print(block,arm,statistics.median(values[4:]),flush=True)
        summary={arm:dict(median_ms=statistics.median(v for r in report['rows'] if r['arm']==arm for v in r['step_ms']),
            p95_ms=float(np.percentile([v for r in report['rows'] if r['arm']==arm for v in r['step_ms']],95))) for arm in ('control','shared_index')}
        blocks=[{arm:statistics.median(v for r in report['rows'] if r['block']==block and r['arm']==arm for v in r['step_ms']) for arm in summary} for block in range(3)]
        saving=summary['control']['median_ms']-summary['shared_index']['median_ms']
        report.update(passed=True,summary=summary,blocks=blocks,median_saving_ms=saving,
            advance_to_model=saving>=4 and all(b['shared_index']<b['control'] for b in blocks) and summary['shared_index']['p95_ms']<=1.02*summary['control']['p95_ms'],
            limitations='Complete post-victim hot compaction only; excludes append/promotion/victim-selection/backup/cold overflow/model. CPUkeepmask/indices/positions, indexupload, allocation and allKV/metadata gathers charged. No full-model or quality claim; no backend edits.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('summary','median_saving_ms','advance_to_model')},indent=2))


if __name__=='__main__':main()
