"""GPU command consumption against original cache, including cold overflow."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from cpu_state import CpuStateArena
from pooled_state import StateArena
from tiered_kv_cache import TieredKVCache,TieredConfig
BASE=Path(__file__).resolve().parent

def pack(rows,dtype=np.int32):
    a=np.full((len(rows),max(map(len,rows),default=0)),-1,dtype=dtype)
    for i,row in enumerate(rows):a[i,:len(row)]=row
    return a,np.asarray(list(map(len,rows)),np.int32)

def bits(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    kind=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
    assert torch.equal(a.contiguous().view(kind),b.contiguous().view(kind))

def payload(shape):return torch.randint(-32768,32767,shape,dtype=torch.int16,device='cuda').view(torch.bfloat16)

@torch.inference_mode()
def run(label,layers=3,heads=2,dim=8):
    inclusive=label!='exclusive';old=8;capacity=12
    if label in ('shadow','group8'):
        positions=[2,5,7,100,101,102];shadow=[True]*3+[False]*3;promotions=[3,5];victims=[2,8,3,4];cold_budget=6;floor=1
    elif label=='floor_cover':
        positions=[100,101,102,103];shadow=[False]*4;promotions=[];victims=[3,4];cold_budget=4;floor=6
    else:
        positions=[100,101,102,103];shadow=[False]*4;promotions=[0,2];victims=[8,9,3,4];cold_budget=4 if inclusive else 2;floor=1
    slot_order=[9,2,10,4,8,5][:len(positions)];free=[i for i in range(capacity) if i not in slot_order]
    cpu=CpuStateArena(layers,capacity,capacity);gpu=StateArena(layers,capacity,capacity,'cuda')
    gpu.hot.fill_(-777);gpu.cold.fill_(-888)
    cpu_roots={k:a.ctypes.data for k,a in cpu.owners.items()};meta_roots=(gpu.hot.data_ptr(),gpu.cold.data_ptr())
    originals=[];states=[];physical=[];raw=[];padding=[]
    for l in range(layers):
        k,v=payload((heads,capacity,dim)),payload((heads,capacity,dim));pk,pv=payload((heads,capacity,dim)),payload((heads,capacity,dim))
        cold_index=torch.tensor(slot_order,device='cuda')
        cfg=TieredConfig(num_heads=heads,head_dim=dim,sram_capacity=old+len(promotions)+1-len(victims),sttram_capacity=cold_budget,stt_live_floor=floor,inclusive=inclusive,device='cuda',kv_dtype=torch.bfloat16)
        c=TieredKVCache(cfg);c._t=7
        c.sram_k,c.sram_v=k[:,:old].clone(),v[:,:old].clone();c.sram_pos=list(range(old));c.sram_grace=torch.zeros(old,dtype=torch.bool)
        cumulative=torch.arange(old,device='cuda',dtype=torch.float32)+l
        cumulative[1]=torch.tensor(-0.,device='cuda');cumulative[5]=torch.tensor(float('nan'),device='cuda')
        c.sram_cum=cumulative.clone();c.sram_age=torch.arange(old,device='cuda',dtype=torch.float32)
        c.stt_k,c.stt_v=pk.index_select(1,cold_index),pv.index_select(1,cold_index);c.stt_pos=positions.copy()
        c.stt_shadow=torch.tensor(shadow);c.stt_was_vram_resident=torch.zeros(len(positions),dtype=torch.bool)
        c.stt_peek_streak=torch.arange(len(positions),dtype=torch.float32)
        c.stt_last_access=torch.tensor([i%2 for i in range(len(positions))],device='cuda',dtype=torch.float32)
        cpu.load(l,c.sram_pos,c.stt_pos,c.sram_grace.numpy(),c.stt_shadow.numpy(),c.stt_was_vram_resident.numpy(),c.stt_peek_streak.numpy(),slot_order,free)
        cum,age,stamp=gpu.attach(l,c.sram_cum,c.sram_age,c.stt_last_access)
        states.append(SimpleNamespace(sram_k=k[:,:old],sram_v=v[:,:old],sram_cum=cum,sram_age=age))
        c._select_hot_victims=lambda:victims.copy()
        originals.append(c);raw.append((k,v));physical.append((pk,pv))
        total=old+len(promotions)+1;padding.append((k[:,total:].clone(),v[:,total:].clone()))
    commands=cpu.promote(*pack([promotions]*layers),inclusive=inclusive,floor=floor)
    cpu.validate_commands(commands)
    gpu.prepare(np.asarray([[old,len(promotions),1]]*layers,dtype=np.int32))
    for l,(c,state,pair,pool) in enumerate(zip(originals,states,raw,physical)):
        promoted=c.promote([(i,i+1) for i in promotions]);assert promoted==len(promotions)
        c.sram_age+=1
        n=int(commands['counts'][l]);start=int(commands['hot_start'][l]);assert start==old
        idx=torch.tensor(commands['read_slots'][l,:n].copy(),device='cuda',dtype=torch.long)
        for target,source in zip(pair,pool):target[:,start:start+n].copy_(source.index_select(1,idx))
        if inclusive:
            gpu.cold[l,commands['rows'][l,:n].tolist()]=7
        else:
            keep=[i for i in range(len(positions)) if i not in promotions]
            selected=gpu.cold[l,:len(positions)][keep].clone();gpu.cold[l,:len(keep)].copy_(selected)
        state.sram_cum=gpu.hot[0,l,:old+n];state.sram_age=gpu.hot[1,l,:old+n]
    cpu.append_hot(*pack([[200]]*layers,dtype=np.int64))
    all_new=[];outcomes=[]
    for l,(c,state,pair) in enumerate(zip(originals,states,raw)):
        new_k,new_v=payload((heads,1,dim)),payload((heads,1,dim));all_new.append((new_k,new_v))
        total=old+len(promotions)+1
        pair[0][:,total-1:total].copy_(new_k);pair[1][:,total-1:total].copy_(new_v)
        state.sram_k,state.sram_v=pair[0][:,:total],pair[1][:,:total]
        state.sram_cum,state.sram_age=gpu.hot[0,l,:total],gpu.hot[1,l,:total]
        outcomes.append(c.evict_and_demote(new_k,new_v,new_pos=200))
    commands=cpu.demote(*pack([victims]*layers),inclusive=inclusive);cpu.validate_commands(commands)
    keeps=[];cold_before=[]
    for l,(state,pool) in enumerate(zip(states,physical)):
        paid,restored,kept=map(int,commands['sizes'][l]);cold_n=int(cpu.owners['cold_n'][l]);old_cold=cold_n-paid
        idx=torch.tensor(commands['paid'][l,:paid].copy(),device='cuda',dtype=torch.long)
        destinations=commands['writes'][l,:paid].tolist()
        # Advanced indexing creates a temporary; write using index_copy_ into the owner.
        for target,source in zip(pool,[state.sram_k,state.sram_v]):target.index_copy_(1,torch.tensor(destinations,device='cuda',dtype=torch.long),source.index_select(1,idx))
        gpu.cold[l,commands['restored'][l,:restored].tolist()]=7
        gpu.cold[l,old_cold:cold_n]=7
        keeps.append(commands['keep'][l,:kept].copy());cold_before.append(cpu.view('cold',l).copy())
    gpu.compact(states,keeps,0)
    # Metadata is now compacted before selecting/applying cold reclamation.
    cold_keeps=[]
    for l,(c,state) in enumerate(zip(originals,states)):
        assert cpu.view('hot',l).tolist()==c.sram_pos
        bits(state.sram_k,c.sram_k);bits(state.sram_v,c.sram_v);bits(state.sram_cum,c.sram_cum);bits(state.sram_age,c.sram_age)
        assert len(set(cold_before[l]))==len(cold_before[l])
        keep=[i for i,p in enumerate(cold_before[l]) if p in set(c.stt_pos)]
        assert cold_before[l][keep].tolist()==c.stt_pos
        cold_keeps.append(keep)
        selected=gpu.cold[l,:len(cold_before[l])][keep].clone();gpu.cold[l,:len(keep)].copy_(selected)
    cpu.reclaim(*pack(cold_keeps))
    try:cpu.validate_commands(commands)
    except ValueError:pass
    else:raise AssertionError('Reclaim did not invalidate old payload commands')
    for l,(c,pool,pair,state) in enumerate(zip(originals,physical,raw,states)):
        assert cpu.view('cold',l).tolist()==c.stt_pos
        for name,field in [('grace','sram_grace'),('shadow','stt_shadow'),('resident','stt_was_vram_resident'),('streak','stt_peek_streak')]:assert cpu.view(name,l).tobytes()==getattr(c,field).numpy().tobytes()
        bits(gpu.cold[l,:len(c.stt_pos)],c.stt_last_access)
        idx=torch.tensor(cpu.view('slots',l).copy(),device='cuda',dtype=torch.long)
        bits(pool[0].index_select(1,idx),c.stt_k);bits(pool[1].index_select(1,idx),c.stt_v)
        total=old+len(promotions)+1;bits(pair[0][:,total:],padding[l][0]);bits(pair[1][:,total:],padding[l][1])
        assert torch.equal(gpu.hot[:,l,total:],torch.full_like(gpu.hot[:,l,total:],-777))
        assert torch.equal(gpu.cold[l,len(cold_before[l]):],torch.full_like(gpu.cold[l,len(cold_before[l]):],-888))
    assert cpu_roots=={k:a.ctypes.data for k,a in cpu.owners.items()} and meta_roots==(gpu.hot.data_ptr(),gpu.cold.data_ptr())
    torch.cuda.synchronize()
    return dict(case=label,layers=layers,heads=heads,dim=dim,outcomes=outcomes,roots_padding_payload_f32_exact=True,cpu_owned_bytes=cpu.owned_bytes,gpu_owned_bytes=gpu.owned_gpu_bytes,descriptor_h2d_bytes=gpu.descriptor_h2d_bytes,scratch_peak_bytes=gpu.scratch_peak_bytes)

def main():
    output=BASE/'gpu.json';assert not output.exists();torch.manual_seed(7)
    rows=[run(case) for case in ['shadow','lru','floor_cover','exclusive']]
    rows.append(run('group8',8,8,128))
    report=dict(passed=True,rows=rows,gpu=torch.cuda.get_device_name(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiments/cpu_state.py',ROOT/'experiments/native_cpu_state.cpp',ROOT/'experiments/pooled_state.py',Path(__file__)]},limits=json.loads((BASE/'gpu_plan.json').read_text())['limits'],full_controller_integrated=False,ordinary_adoption=False,full_goal_complete=False)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
