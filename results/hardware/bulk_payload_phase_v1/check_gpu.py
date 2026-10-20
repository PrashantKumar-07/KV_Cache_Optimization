"""Bit-exact supplied-decision GPU primitive and atomic rejection checks."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import torch
from bulk_payload import execute,kernel
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def bits_equal(a,b):return a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def fixture(layers=9,heads=2,dim=16,old=8,exhaustive=False):
    states=[];keeps=[];paid=[];restored=[]
    for i in range(layers):
        prom=0 if exhaustive else i%3
        n=old+prom+1;pitch=n+3
        roots=[torch.randint(-32768,32768,(heads,pitch,dim),device='cuda',dtype=torch.int16).view(torch.bfloat16) for _ in range(2)]
        promotion=[torch.randint(-32768,32768,(heads,prom,dim),device='cuda',dtype=torch.int16).view(torch.bfloat16) for _ in range(2)]
        new=[root[:,n-1:n] for root in roots]
        if exhaustive:
            for value in new:value.copy_(torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1,1,-1))
        p=[n-1,1] if n>2 else [n-1];k=[n-1] if exhaustive else list(reversed(range(min(5,n))))
        cold=5;r=[1,3];meta=[torch.randint(-2147483648,2147483647,(n,),device='cuda',dtype=torch.int32).view(torch.float32) for _ in range(2)]
        stamp=torch.randint(-2147483648,2147483647,(cold+len(p),),device='cuda',dtype=torch.int32).view(torch.float32)
        states.append(SimpleNamespace(old_k=roots[0][:,:old],old_v=roots[1][:,:old],prom_k=promotion[0],prom_v=promotion[1],new_k=new[0],new_v=new[1],cumulative=meta[0],age=meta[1],timestamps=stamp,old_cold=cold,time=16384+i))
        keeps.append(k);paid.append(p);restored.append(r)
    return states,keeps,paid,restored

def reference(states,keeps,paid,restored):
    views=[];packed=[];times=[]
    for s,k,p,r in zip(states,keeps,paid,restored):
        channels=[torch.cat((old,prom,new.float().bfloat16()),dim=1) for old,prom,new in [(s.old_k,s.prom_k,s.new_k),(s.old_v,s.prom_v,s.new_v)]]
        views.append(tuple(t[:,k].clone() for t in channels)+(s.cumulative[k].clone(),s.age[k].clone()))
        packed.append(torch.stack([t[:,p] for t in channels]).reshape(2,-1,1,channels[0].shape[-1]))
        time=s.timestamps.clone();time[r]=s.time;time[s.old_cold:s.old_cold+len(p)]=s.time;times.append(time)
    return views,torch.cat(packed,dim=1),times

@torch.inference_mode()
def main():
    output=BASE/'gpu.json';assert not output.exists();torch.manual_seed(0);report=dict(passed=False,checks=[],rejections=[],plan_sha256=sha(BASE/'plan.json'))
    try:
        for name,args in [('ragged9',{}),('exhaustive_new_BF16',dict(layers=1,heads=1,dim=65536,old=1,exhaustive=True)),('single_layer',dict(layers=1)),('full32',dict(layers=32,heads=8,dim=128,old=649))]:
            states,k,p,r=fixture(**args);views,paid,times=reference(states,k,p,r);actual=execute(states,k,p,r)
            assert all(bits_equal(a,b) for va,vb in zip(actual['views'],views) for a,b in zip(va,vb)),name
            assert bits_equal(actual['paid'],paid),name
            assert all(bits_equal(s.timestamps,t) for s,t in zip(states,times)),name
            report['checks'].append(dict(case=name,exact=True,layers=len(states),descriptor_bytes=actual['descriptor_h2d_bytes'],scratch_peak_bytes=actual['scratch_peak_bytes'],paid_buffer_bytes=actual['paid_buffer_bytes']))
        for name in ['bad_later_keep','cross_layer_new_alias','metadata_overlap','timestamp_overflow','hot_capacity','empty_hot']:
            states,k,p,r=fixture(layers=1,old=0) if name=='empty_hot' else fixture();targets=[[s.old_k,s.old_v,s.prom_k,s.prom_v,s.new_k,s.new_v,s.cumulative,s.age,s.timestamps] for s in states];before=[[t.clone() for t in values] for values in targets]
            if name=='bad_later_keep':k[-1]=[-1]
            if name=='cross_layer_new_alias':states[-1].new_k=states[0].old_k[:,:1];states[-1].new_v=states[0].old_v[:,:1]
            if name=='metadata_overlap':states[-1].age=states[-1].cumulative
            if name=='timestamp_overflow':states[-1].time=2**24+1
            if name=='hot_capacity':states[-1].old_k=states[-1].old_k.clone();states[-1].old_v=states[-1].old_v.clone()
            try:execute(states,k,p,r)
            except ValueError as error:
                if name=='empty_hot':assert 'Empty hot payload' in str(error)
            else:raise AssertionError(name+' not rejected')
            for values,original in zip(targets,before):
                assert all(bits_equal(value,expected) for value,expected in zip(values,original)),name
            report['rejections'].append(dict(case=name,rejected_before_mutation=True))
        binary=Path(kernel().__file__);report.update(passed=True,binary_path=str(binary),binary_sha256=sha(binary),full_goal_complete=False,policy_integrated=False,limits='Supplied decisions and standalone GPU payload/metadata roots. No selector/CPUownership/nativehosttransfer/modelspeed or complete512MiB proof.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:output.write_text(json.dumps(report,indent=2)+'\n')
    print('Bulk payload bit parity and malformed-input atomicity pass',flush=True)
if __name__=='__main__':main()
