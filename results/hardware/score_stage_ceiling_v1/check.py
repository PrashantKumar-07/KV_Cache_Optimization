"""Optimistic ceiling screen: charge current scoring versus zero work."""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import numpy as np
import torch
from batched_score_policy import batched_update
from cpu_phase_policy import CpuPhaseCache
from cpu_state import CpuStateArena
from pooled_state import StateArena
BASE=Path(__file__).resolve().parent
class Fixture(CpuPhaseCache):
    def __del__(self):pass

def fixture(hot,cold,mode):
    cpu=CpuStateArena(32,hot+33,cold+64);gpu=StateArena(32,hot+33,cold+64,'cuda')
    gpu.hot.zero_();gpu.cold.zero_();caches={};positions={};indices={}
    for layer in range(32):
        cpu.load(layer,np.arange(hot),np.arange(10000,10000+cold),np.zeros(hot,bool),np.zeros(cold,bool),np.zeros(cold,bool),np.zeros(cold,np.float32),np.arange(cold),np.arange(cold,cold+64))
        c=Fixture.__new__(Fixture);c._cpu_arena=cpu;c._cpu_layer=layer;c._cpu_cached_version=-1;c._cpu_cached={};c._state_arena=gpu;c._state_layer=layer;c._sram_cum=gpu.hot[0,layer,:hot];c._sram_age=gpu.hot[1,layer,:hot];c._stt_last_access=gpu.cold[layer,:cold];c._t=16384+layer;caches[layer]=c
        indices[layer]=np.linspace(0,cold-1,32,dtype=np.int64).tolist();positions[layer]=[c.stt_pos[i] for i in indices[layer]]
    policy=SimpleNamespace(layer_caches=caches,_vis_sram_keep={i:list(range(hot)) for i in caches},_vis_stt_pos=positions,_vis_stt_idx=indices)
    weights=[torch.rand((1,32,1,hot+33),device='cuda',dtype=torch.bfloat16) for _ in caches]
    if mode=='zero':
        for w in weights:w.zero_()
    if mode=='sparse':
        for w in weights:w[:,:,:,hot::2]=0
    return policy,[None]*32,weights,gpu

@torch.inference_mode()
def main():
    out=BASE/'screen.json';assert not out.exists();torch.manual_seed(0)
    report=dict(passed=False,rows=[],semantic_checks=[],plan_sha256=hashlib.sha256((BASE/'plan.json').read_bytes()).hexdigest())
    try:
        for hot,cold in [(649,1982),(991,1090)]:
            for mode in ['positive','zero','sparse']:
                p,past,weights,gpu=fixture(hot,cold,mode)
                plain={i:SimpleNamespace(sram_cum=c.sram_cum.clone(),stt_last_access=c.stt_last_access.clone(),stt_pos=list(c.stt_pos),_t=c._t) for i,c in p.layer_caches.items()}
                ref=SimpleNamespace(**{**vars(p),'layer_caches':plain})
                batched_update(p,past,weights);batched_update(ref,past,weights)
                assert p._stt_attn==ref._stt_attn and p._stt_attn_idx==ref._stt_attn_idx
                assert all(torch.equal(c.sram_cum.view(torch.int32),plain[i].sram_cum.view(torch.int32)) and torch.equal(c.stt_last_access.view(torch.int32),plain[i].stt_last_access.view(torch.int32)) for i,c in p.layer_caches.items())
                report['semantic_checks'].append(dict(hot=hot,cold=cold,mode=mode,exact=True))
                if mode!='positive':continue
                for block in range(3):
                    for arm in ['current','zero_work','zero_work','current']:
                        values=[]
                        for step in range(24):
                            torch.cuda.synchronize();start=time.perf_counter()
                            if arm=='current':batched_update(p,past,weights)
                            torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
                        report['rows'].append(dict(hot=hot,cold=cold,block=block,arm=arm,step_ms=values[4:],warmup_ms=values[:4]))
            print(hot,cold,'finished',flush=True)
        summary=[]
        for hot,cold in [(649,1982),(991,1090)]:
            medians={arm:float(np.median([v for row in report['rows'] if row['hot']==hot and row['arm']==arm for v in row['step_ms']])) for arm in ['current','zero_work']}
            ceiling=medians['current']-medians['zero_work'];summary.append(dict(hot=hot,cold=cold,median_ms=medians,optimistic_saving_ms=ceiling,can_meet_4ms_screen=ceiling>=4))
        report.update(passed=True,summary=summary,full_goal_complete=False,limitation='Synthetic current canonical CPU properties/pooled GPU metadata,32layers/32heads/32exposedrows. Optimistic zero-work arm has no semantics and is never a candidate. No model/generation/resource/quality/physicalSTT proof. Synthetic stage ceiling does not bound all real model trajectories; it cheaply rejects this isolated mechanism at these representative geometries.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
