"""Standalone checked GPU payload phase; fixed decisions, no policy adoption."""
from functools import lru_cache
import hashlib,os
from pathlib import Path
import numpy as np
import torch

@lru_cache(maxsize=1)
def kernel():
    from torch.utils.cpp_extension import load
    root=Path(__file__).parent;files=[root/'native_bulk_payload.cpp',root/'native_bulk_payload.cu']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    build=Path('/data/nishant/tieredkv_external')/('bulk_payload_'+digest[:16]);build.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    return load(name='tieredkv_bulk_payload_v1',sources=list(map(str,files)),build_directory=str(build),extra_cflags=['-O2'],extra_cuda_cflags=['-O3','--fmad=false'])

def indices(value,limit):
    a=np.asarray(value)
    if a.ndim!=1 or (a.size and (not np.issubdtype(a.dtype,np.integer) or np.any(a<0) or np.any(a>=limit))):raise ValueError('Invalid payload indices')
    return np.asarray(a,dtype=np.int64)

def execute(states,keeps,paid,restored):
    """State fields: old/prom/new K,V; cumulative/age; timestamps; old_cold/time.

    Mutates old hot roots, cumulative/age prefixes and timestamps. Returns
    rebound views and packed paid KV. All layers validate before any mutation.
    Caller retains responsibility for CPU ownership, decisions and transfers.
    """
    if not states or len(states)!=len(keeps) or len(states)!=len(paid) or len(states)!=len(restored):raise ValueError('Invalid payload batch')
    first=states[0].old_k
    if first.ndim!=3:raise ValueError('Invalid hot payload rank')
    heads,_,dim=first.shape;device=first.device
    if device.type!='cuda' or heads<1 or dim<1:raise ValueError('CUDA payload geometry required')
    payload_roots=[];metadata_ranges=[];validated=[];sources=[];total_paid=0
    for s,k,p,r in zip(states,keeps,paid,restored):
        values=[s.old_k,s.old_v,s.prom_k,s.prom_v,s.new_k,s.new_v]
        if any(t.ndim!=3 or t.shape[0]!=heads or t.shape[2]!=dim or t.dtype!=torch.bfloat16 or t.device!=device or t.requires_grad or t.stride(2)!=1 or t.stride(1)!=dim or t.stride(0)<t.shape[1]*dim or t.stride(0)%dim for t in values):raise ValueError('Unsupported payload layout')
        if any(values[j].shape!=values[j+1].shape or values[j].stride()!=values[j+1].stride() for j in [0,2,4]) or s.new_k.shape[1]!=1:raise ValueError('Unequal channel geometry')
        old=s.old_k.shape[1];prom=s.prom_k.shape[1];n=old+prom+1
        if old<1:raise ValueError('Empty hot payload requires caller fallback')
        if any(t.stride(0)<n*dim or t.storage_offset()+(heads-1)*t.stride(0)+n*dim>t.untyped_storage().nbytes()//t.element_size() for t in values[:2]):raise ValueError('Hot root lacks reserved capacity')
        k,p=indices(k,n),indices(p,n)
        if type(s.old_cold) is not int or s.old_cold<0 or type(s.time) is not int or not 0<=s.time<=2**24:raise ValueError('Invalid timestamp contract')
        r=indices(r,s.old_cold)
        metadata=[s.cumulative,s.age,s.timestamps]
        if any(t.ndim!=1 or t.dtype!=torch.float32 or t.device!=device or t.requires_grad or not t.is_contiguous() for t in metadata) or s.cumulative.shape!=(n,) or s.age.shape!=(n,) or len(s.timestamps)<s.old_cold+len(p):raise ValueError('Invalid metadata geometry')
        for t in metadata:metadata_ranges.append((t.data_ptr(),t.data_ptr()+t.numel()*4))
        payload_roots.append([t.untyped_storage()._cdata for t in values[:2]])
        sources.append(values+metadata)
        validated.append((s,k,p,r,old,prom,total_paid));total_paid+=heads*len(p)
    roots=[root for pair in payload_roots for root in pair]
    if len(set(roots))!=len(roots):raise ValueError('Hot payload roots must be unique')
    for i,values in enumerate(sources):
        forbidden=set(roots)-set(payload_roots[i])
        if any(t.untyped_storage()._cdata in forbidden for t in values[2:6]) or any(t.untyped_storage()._cdata in roots for t in values[6:]):raise ValueError('Payload or metadata aliases another destination')
    ranges=sorted(metadata_ranges)
    if any(b[0]<a[1] for a,b in zip(ranges,ranges[1:])):raise ValueError('Metadata ranges overlap')
    kw=max(map(len,keeps));pw=max(map(len,paid));rw=max(map(len,restored));desc=np.zeros((len(states),20+kw+pw+rw),dtype=np.int64)
    for i,(s,k,p,r,old,prom,base) in enumerate(validated):
        desc[i,:9]=[t.data_ptr() for t in sources[i]]
        desc[i,9:20]=[old,prom,s.old_k.stride(0)//dim,s.prom_k.stride(0)//dim,s.new_k.stride(0)//dim,len(p),len(r),len(k),base,s.old_cold,s.time]
        desc[i,20:20+len(k)]=k;desc[i,20+kw:20+kw+len(p)]=p;desc[i,20+kw+pw:20+kw+pw+len(r)]=r
    descriptors=torch.from_numpy(desc).to(device);packed=torch.empty((2,total_paid,1,dim),dtype=torch.bfloat16,device=device)
    scratch_peak=0;stream=torch.cuda.current_stream(device)
    for values in sources:
        for t in values:t.record_stream(stream)
    for start in range(0,len(states),8):
        group=min(8,len(states)-start);hot=torch.empty((2,group,heads,kw,dim),dtype=torch.bfloat16,device=device);meta=torch.empty((2,group,kw),dtype=torch.float32,device=device)
        scratch_peak=max(scratch_peak,hot.numel()*2+meta.numel()*4)
        kernel().run(descriptors,hot,meta,packed,start,kw,pw,rw)
        del hot,meta
    views=[]
    for s,k,_,_,_,_,_ in validated:
        views.append(tuple(torch.as_strided(t,(heads,len(k),dim),t.stride(),t.storage_offset()) for t in [s.old_k,s.old_v])+(s.cumulative[:len(k)],s.age[:len(k)]))
    return dict(views=views,paid=packed,descriptor_h2d_bytes=desc.nbytes,scratch_peak_bytes=scratch_peak,paid_buffer_bytes=packed.numel()*2)
