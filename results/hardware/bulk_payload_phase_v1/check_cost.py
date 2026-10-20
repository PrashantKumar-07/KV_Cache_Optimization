"""Charge fused phase against current append/pack/touch/bounded compact."""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import numpy as np
import torch
from bulk_payload import execute,kernel
from raw_input_reuse import RawInputCache
from pooled_state import StateArena,kernel as pooled_kernel
from pooled_state_policy import PooledCache
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def fixture(hot,cold,all_paid):
    torch.manual_seed(0);arena=StateArena(32,hot+64,cold+64,'cuda');arena.hot.zero_();arena.cold.zero_()
    states=[];keeps=[];paid=[];restored=[]
    for i in range(32):
        n=hot+33;roots=[torch.randint(-32768,32768,(8,n,128),device='cuda',dtype=torch.int16).view(torch.bfloat16) for _ in range(2)]
        promotion=[torch.randint(-32768,32768,(8,32,128),device='cuda',dtype=torch.int16).view(torch.bfloat16) for _ in range(2)]
        p=list(range(33)) if all_paid else [32];r=[] if all_paid else list(range(32));old_cold=cold-400
        states.append(SimpleNamespace(old_k=roots[0][:,:hot],old_v=roots[1][:,:hot],prom_k=promotion[0],prom_v=promotion[1],new_k=roots[0][:,n-1:n],new_v=roots[1][:,n-1:n],cumulative=arena.hot[0,i,:n],age=arena.hot[1,i,:n],timestamps=arena.cold[i,:old_cold+len(p)],old_cold=old_cold,time=16384+i))
        keeps.append(list(range(33,n)));paid.append(p);restored.append(r)
    return states,keeps,paid,restored,arena

def reference(states,keeps,paid,restored,arena):
    caches=[];packed=[]
    for i,(s,k,p,r) in enumerate(zip(states,keeps,paid,restored)):
        h,old,d=s.old_k.shape;pitch=s.old_k.stride(0)//d
        raw=[torch.as_strided(t,(1,h,pitch,d),(h*t.stride(0),*t.stride()),t.storage_offset()) for t in [s.old_k,s.old_v]]
        c=SimpleNamespace(sram_k=s.old_k,sram_v=s.old_v,sram_cum=s.cumulative[:old],sram_age=s.age[:old],stt_last_access=s.timestamps[:s.old_cold],_state_arena=arena,_state_layer=i,_pool_prepared=True,_hot_buffer=raw,_t=s.time)
        # Exact existing per-layer payload contracts and metadata view extension.
        RawInputCache._extend_hot_pair(c,s.prom_k,s.prom_v)
        c.sram_cum=PooledCache.extend_metadata(c,'sram_cum',32);c.sram_age=PooledCache.extend_metadata(c,'sram_age',32)
        new_k,new_v=s.new_k.float(),s.new_v.float()
        RawInputCache._extend_hot_pair(c,new_k.bfloat16(),new_v.bfloat16())
        c.sram_cum=PooledCache.extend_metadata(c,'sram_cum',1);c.sram_age=PooledCache.extend_metadata(c,'sram_age',1)
        # Same GPU paid gather/stack payload as SharedPool.enqueue. CPU slot
        # conflict/address bookkeeping and physical transfers excluded in BOTH.
        packed.append(torch.stack((c.sram_k[:,p],c.sram_v[:,p])).to(torch.bfloat16).contiguous().reshape(2,-1,1,d))
        c.stt_last_access[r]=s.time;c.stt_last_access=PooledCache.append_timestamps(c,len(p));caches.append(c)
    result=torch.cat(packed,dim=1)
    for first in range(0,len(caches),8):arena.compact(caches[first:first+8],keeps[first:first+8],first)
    return dict(views=[(c.sram_k,c.sram_v,c.sram_cum,c.sram_age) for c in caches],paid=result)

def equal(a,b):return a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def compare(a,b,sa,sb):
    assert all(equal(x,y) for va,vb in zip(a['views'],b['views']) for x,y in zip(va,vb))
    assert equal(a['paid'],b['paid']) and all(equal(x.timestamps,y.timestamps) for x,y in zip(sa,sb))

@torch.inference_mode()
def main():
    output=BASE/'cost.json';assert not output.exists();report=dict(passed=False,rows=[],summary=[],plan_sha256=sha(BASE/'cost_plan.json'))
    try:
        kernel();pooled_kernel()
        for hot,cold,all_paid in [(649,1982,False),(991,1090,False),(649,1982,True)]:
            current=fixture(hot,cold,all_paid);bulk=fixture(hot,cold,all_paid)
            a=reference(*current);b=execute(*bulk[:4]);torch.cuda.synchronize();compare(a,b,current[0],bulk[0]);a=b=None
            for block in range(3):
                for arm in ['current','bulk','bulk','current']:
                    samples=[]
                    for step in range(24):
                        torch.cuda.synchronize();start=time.perf_counter()
                        result=reference(*current) if arm=='current' else execute(*bulk[:4])
                        torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000);result=None
                    report['rows'].append(dict(hot=hot,cold=cold,all_paid=all_paid,block=block,arm=arm,warmup_ms=samples[:4],step_ms=samples[4:]))
            a=reference(*current);b=execute(*bulk[:4]);torch.cuda.synchronize();compare(a,b,current[0],bulk[0])
            medians={};p95={}
            for arm in ['current','bulk']:
                values=[t for row in report['rows'] if (row['hot'],row['all_paid'],row['arm'])==(hot,all_paid,arm) for t in row['step_ms']]
                medians[arm]=float(np.median(values));p95[arm]=float(np.percentile(values,95))
            saved=medians['current']-medians['bulk'];guard=p95['bulk']<=1.02*p95['current']
            report['summary'].append(dict(hot=hot,cold=cold,all_paid=all_paid,exact=True,median_ms=medians,p95_ms=p95,saved_ms=saved,advance=saved>=4 and guard,scratch_peak_bytes=b['scratch_peak_bytes'],paid_buffer_bytes=b['paid_buffer_bytes'],descriptor_h2d_bytes=b['descriptor_h2d_bytes']))
            print(hot,cold,all_paid,medians,'saved',saved,flush=True)
        report.update(passed=True,advance_to_model=all(v['advance'] for v in report['summary']),full_goal_complete=False,limits='Supplied promotion/keep/paid/restored decisions; CPUownership/slotconflicts/addresses and physical transfers excluded in BOTH. Current payload methods and group8 native compactor invoked. Descriptors/layout/ownership checks, allocations, GPU work, stream recording, output packing/rebinding and synchronization charged. No model/selection/quality/fullmemory/adoption proof.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
