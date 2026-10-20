"""Pooled primitive bitwise, root, padding and stream proof; not model timing."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from pooled_state import StateArena,kernel
BASE=Path(__file__).resolve().parent
bits=lambda t:t.contiguous().view(torch.uint8)
def main():
    output=BASE/'gpu.json';assert not output.exists();rows=[];torch.manual_seed(0)
    for layers,capacity in [(3,16),(32,1055)]:
        arena=StateArena(layers,capacity,1154,'cuda')
        arena.hot.copy_(torch.randint(-2**31,2**31-1,arena.hot.shape,device='cuda',dtype=torch.int32).view(torch.float32))
        before=arena.hot.clone();expected=before.clone();counts=[]
        for layer in range(layers):
            old=min(capacity-4,991);prom=(layer%3);new=1
            counts.append([old,prom,new]);expected[0,layer,old:old+prom+new]=0
            expected[1,layer,:old]=before[1,layer,:old]+1
            expected[1,layer,old:old+prom]=1;expected[1,layer,old+prom:old+prom+new]=0
        root=arena.hot.untyped_storage()._cdata;arena.prepare(counts);torch.cuda.synchronize()
        assert torch.equal(bits(arena.hot),bits(expected)) and arena.hot.untyped_storage()._cdata==root
        rows.append(dict(case='prepare',layers=layers,exact=True,descriptor_bytes=arena.descriptor_h2d_bytes,owned_bytes=arena.owned_gpu_bytes))
    for count,heads,old,dim,keep_rows,worker in [(1,2,14,4,7,False),(3,2,14,4,7,True),(8,8,1024,128,991,False)]:
        arena=StateArena(count,old+32,1154,'cuda');arena.hot.fill_(17);states=[];keeps=[];expected=[];owners=[]
        for layer in range(count):
            full=[torch.randint(-32768,32767,(heads,old+16,dim),device='cuda',dtype=torch.int16).view(torch.bfloat16) for _ in range(2)]
            cum=torch.randint(-2**31,2**31-1,(old,),device='cuda',dtype=torch.int32).view(torch.float32)
            age=torch.randint(-2**31,2**31-1,(old,),device='cuda',dtype=torch.int32).view(torch.float32)
            c,a,t=arena.attach(layer,cum,age,torch.zeros(20,device='cuda'))
            state=SimpleNamespace(sram_k=full[0][:,:old],sram_v=full[1][:,:old],sram_cum=c,sram_age=a)
            keep=np.sort(np.random.default_rng(layer).choice(old,keep_rows,replace=False)).astype(np.int32);index=torch.as_tensor(keep.astype(np.int64),device='cuda')
            expected.append([state.sram_k.index_select(1,index),state.sram_v.index_select(1,index),c[index],a[index]])
            keeps.append(keep);states.append(state);owners.append([x.untyped_storage()._cdata for x in full])
        padding=arena.hot[:,:,keep_rows:].clone();arena_root=arena.hot.untyped_storage()._cdata
        if worker:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):arena.compact(states,keeps,0)
            torch.cuda.current_stream().wait_stream(stream)
        else:arena.compact(states,keeps,0)
        torch.cuda.synchronize()
        for state,want,roots in zip(states,expected,owners):
            for actual,value in zip([state.sram_k,state.sram_v,state.sram_cum,state.sram_age],want):assert torch.equal(bits(actual),bits(value))
            assert [x.untyped_storage()._cdata for x in [state.sram_k,state.sram_v]]==roots
            assert state.sram_cum.untyped_storage()._cdata==state.sram_age.untyped_storage()._cdata==arena_root
        assert torch.equal(bits(arena.hot[:,:,keep_rows:]),bits(padding))
        rows.append(dict(case='compact',layers=count,heads=heads,rows=keep_rows,exact=True,nondefault_stream=worker,descriptor_bytes=arena.descriptor_h2d_bytes,scratch_peak_bytes=arena.scratch_peak_bytes,roots_preserved=True,padding_preserved=True))
    arena=StateArena(1,16,16,'cuda');rejected=0
    for counts in [torch.zeros((1,3),dtype=torch.float64,device='cuda'),torch.zeros((1,2),dtype=torch.int32,device='cuda')]:
        try:kernel().prepare(arena.hot,counts)
        except RuntimeError:rejected+=1
    assert rejected==2
    binary=Path(kernel().__file__)
    report=dict(passed=True,rows=rows,malformed_native_descriptor_guards=rejected,binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='Pooled GPU primitive exact bits/views/roots/padding/workerstream. No fullmodel controller integration, wholeloop/resources/quality/physicalSTT/adoption proof.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(rows),guards=rejected,rows=rows)))
if __name__=='__main__':main()
