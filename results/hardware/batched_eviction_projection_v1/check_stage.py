"""Charged complete projected-state preparation plus original victim selection."""
import hashlib,json,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from batched_eviction_projection import PROJECTED_VICTIMS,projected_scores,COORDINATOR_AST,SELECTION_AST
from probe_batched_eviction_selection import batched_victims,cache
from provenance import atomic_json,runtime_identity
import importlib.util
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quiet_guard',ROOT/'results/hardware/pool_query_selector_v1/check_timing.py')
quiet_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(quiet_module)


def prepare(caches,n_new,fast):
    projections=[]
    for c in caches:
        positions=c.sram_pos+list(range(16384,16384+n_new))
        grace=torch.cat((c.sram_grace,torch.zeros(n_new,dtype=torch.bool)))
        if fast:cum,age=c.sram_cum,c.sram_age
        else:
            zeros=torch.zeros(n_new,device=c.sram_cum.device,dtype=c.sram_cum.dtype)
            cum=torch.cat((c.sram_cum,zeros));age=torch.cat((c.sram_age+1,zeros))
        projections.append(SimpleNamespace(cfg=c.cfg,sram_pos=positions,sram_grace=grace,sram_cum=cum,sram_age=age))
    return PROJECTED_VICTIMS(projections,numpy_masks=True,n_new=n_new) if fast else batched_victims(projections,numpy_masks=True)


def main():
    output=BASE/'stage.json';assert not output.exists()
    report=dict(passed=False,rows=[],checks=[],quiet_observations=[],runtime=runtime_identity(),plan_sha256=hashlib.sha256((BASE/'plan.json').read_bytes()).hexdigest(),coordinator_ast=COORDINATOR_AST,selection_ast=SELECTION_AST)
    try:
        torch.manual_seed(0);obs=quiet_module.quiet();assert obs['quiet'];report['quiet_observations'].append(obs)
        cases=[]
        for i in range(32):
            positions=list(range(4))+list(range(1000+i*13,1859+i*13))+list(range(16256,16384))+list(range(5000+i*33,5032+i*33))
            cum=torch.randint(0,64,(1023,),device='cuda').float()/64
            age=torch.randint(0,64,(1023,),device='cuda').float()
            grace=torch.zeros(1023,dtype=torch.bool);grace[991:]=True
            cases.append(cache(positions,cum,age,capacity=991,grace=grace))
        for n_new in (1,4,32):
            assert prepare(cases,n_new,True)==prepare(cases,n_new,False)
            report['checks'].append(dict(case='primary32',n_new=n_new,victims_exact=True))
        ragged=[]
        for i,size in enumerate([0,1,9,33,64,1057]):
            positions=list(range(size));cum=torch.ones(size,device='cuda');age=torch.zeros(size,device='cuda')
            if size>=9:cum[4:9]=torch.tensor([float('nan'),float('inf'),-float('inf'),-0.,0.],device='cuda')
            grace=torch.ones(size,dtype=torch.bool)
            ragged.append(cache(positions,cum,age,max(0,size-17),grace=grace,window=8))
        for n_new in (1,4,32):
            assert prepare(ragged,n_new,True)==prepare(ragged,n_new,False)
            a,b=projected_scores(ragged,n_new)
            for i,c in enumerate(ragged):
                expected_c=torch.cat((c.sram_cum,torch.zeros(n_new,device='cuda')))
                expected_a=torch.cat((c.sram_age+1,torch.zeros(n_new,device='cuda')))
                assert torch.equal(a[i,:len(expected_c)].view(torch.int32),expected_c.view(torch.int32))
                assert torch.equal(b[i,:len(expected_a)].view(torch.int32),expected_a.view(torch.int32))
            report['checks'].append(dict(case='ragged_nonfinite_grace',n_new=n_new,victims_and_projection_bits_exact=True))
        for block in range(3):
            for arm in ('control','candidate','candidate','control'):
                obs=quiet_module.quiet();assert obs['quiet'];report['quiet_observations'].append(obs)
                values=[]
                for _ in range(24):
                    torch.cuda.synchronize();start=time.perf_counter()
                    selected=prepare(cases,1,arm=='candidate')
                    torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
                report['rows'].append(dict(block=block,arm=arm,warmup_ms=values[:4],step_ms=values[4:]))
                print(block,arm,statistics.median(values[4:]),flush=True)
                obs=quiet_module.quiet();assert obs['quiet'];report['quiet_observations'].append(obs)
        summary={arm:dict(median_ms=statistics.median([v for r in report['rows'] if r['arm']==arm for v in r['step_ms']]),p95_ms=float(np.quantile([v for r in report['rows'] if r['arm']==arm for v in r['step_ms']],.95))) for arm in ('control','candidate')}
        blocks=[{arm:statistics.median([v for r in report['rows'] if r['arm']==arm and r['block']==b for v in r['step_ms']]) for arm in summary} for b in range(3)]
        saving=summary['control']['median_ms']-summary['candidate']['median_ms']
        guards=dict(saved_ms=saving>=4,all_blocks_faster=all(r['candidate']<r['control'] for r in blocks),p95=summary['candidate']['p95_ms']<=1.02*summary['control']['p95_ms'])
        report.update(passed=True,summary=summary,blocks=blocks,median_saving_ms=saving,guards=guards,charged_stage_primary_passed=all(guards.values()),limits='Synthetic complete projection+victim stage after promotions; CPUpositions/grace and CUDApads/age/masks/copies/stablesort/fallback/returns charged. Excludes promotion/write/model. No wholeloop/quality/memory/adoption claim.')
    except BaseException as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(output,report)
if __name__=='__main__':main()
