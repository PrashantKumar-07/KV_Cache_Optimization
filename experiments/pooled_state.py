"""Research fixed metadata owners and checked fused GPU state transitions."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import numpy as np
import torch

@lru_cache(maxsize=1)
def kernel():
    from torch.utils.cpp_extension import load
    source=Path(__file__).parent
    digest=hashlib.sha256(b''.join((source/f).read_bytes() for f in ['native_pooled_state.cpp','native_pooled_state.cu'])).hexdigest()
    build=Path('/data/nishant/tieredkv_external')/('pooled_state_'+digest[:16]);build.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    return load(name='tieredkv_pooled_state_v1',sources=[str(source/'native_pooled_state.cpp'),str(source/'native_pooled_state.cu')],build_directory=str(build),extra_cflags=['-O2'],extra_cuda_cflags=['-O3','--fmad=false'])

class StateArena:
    def __init__(self,layers,hot_capacity,cold_capacity,device):
        if any(type(n) is not int or n<1 for n in [layers,hot_capacity,cold_capacity]):raise ValueError('Positive arena geometry required')
        self.hot=torch.empty((2,layers,hot_capacity),dtype=torch.float32,device=device)
        self.cold=torch.empty((layers,cold_capacity),dtype=torch.float32,device=device)
        self.descriptor_h2d_bytes=0;self.scratch_peak_bytes=0
    def attach(self,layer,cumulative,age,timestamps):
        if (type(layer) is not int or not 0<=layer<self.hot.shape[1] or cumulative.ndim!=1 or cumulative.shape!=age.shape
                or timestamps.ndim!=1 or len(cumulative)>self.hot.shape[2] or len(timestamps)>self.cold.shape[1]
                or any(t.dtype!=torch.float32 or t.device!=self.hot.device or t.requires_grad for t in [cumulative,age,timestamps])):
            raise ValueError('Invalid arena attachment')
        self.hot[0,layer,:len(cumulative)].copy_(cumulative);self.hot[1,layer,:len(age)].copy_(age)
        self.cold[layer,:len(timestamps)].copy_(timestamps)
        return self.hot[0,layer,:len(cumulative)],self.hot[1,layer,:len(age)],self.cold[layer,:len(timestamps)]
    def prepare(self,counts):
        values=np.asarray(counts)
        if values.shape!=(self.hot.shape[1],3) or not np.issubdtype(values.dtype,np.integer) or np.any(values<0) or np.any(values>self.hot.shape[2]) or np.any(values.sum(axis=1)>self.hot.shape[2]):
            raise ValueError('Prepare rows exceed metadata arena')
        packed=np.asarray(values,dtype=np.int32);gpu=torch.from_numpy(packed).to(self.hot.device)
        self.descriptor_h2d_bytes+=packed.nbytes
        self.hot.record_stream(torch.cuda.current_stream(self.hot.device))
        kernel().prepare(self.hot,gpu)
    def compact(self,states,keeps,first):
        if not states or len(states)>8 or len(states)!=len(keeps) or type(first) is not int or first<0 or first+len(states)>self.hot.shape[1]:
            raise ValueError('Invalid arena compaction group')
        h,_,d=states[0].sram_k.shape;rows=len(keeps[0]);indices=np.asarray(keeps)
        if indices.shape!=(len(states),rows) or not np.issubdtype(indices.dtype,np.integer) or rows>self.hot.shape[2]:raise ValueError('Invalid keep geometry')
        owners=[];pitches=[]
        for index,(state,keep) in enumerate(zip(states,keeps)):
            k,v=state.sram_k,state.sram_v;n=k.shape[1]
            if (k.ndim!=3 or k.shape!=(h,n,d) or v.shape!=k.shape or state.sram_cum.shape!=state.sram_age.shape or state.sram_cum.shape!=(n,)
                    or any(t.device!=self.hot.device or t.requires_grad for t in [k,v,state.sram_cum,state.sram_age])
                    or k.dtype!=torch.bfloat16 or v.dtype!=k.dtype or state.sram_cum.dtype!=torch.float32 or state.sram_age.dtype!=torch.float32
                    or any(t.stride(2)!=1 or t.stride(1)!=d or t.stride(0)<n*d for t in [k,v])
                    or rows>n or k.stride(0)%d or k.stride()!=v.stride() or not state.sram_cum.is_contiguous() or not state.sram_age.is_contiguous()
                    or np.any(np.asarray(keep)<0) or np.any(np.asarray(keep)>=n)):
                raise ValueError('Unsupported pooled compaction layout or indices')
            for channel,name in enumerate(['sram_cum','sram_age']):
                t=getattr(state,name)
                if t.untyped_storage()._cdata==self.hot.untyped_storage()._cdata and t.data_ptr()!=self.hot[channel,first+index].data_ptr():
                    raise ValueError('Metadata aliases another arena row')
            if self.hot.untyped_storage()._cdata in [k.untyped_storage()._cdata,v.untyped_storage()._cdata]:raise ValueError('KV aliases metadata arena')
            owners.extend([k.untyped_storage()._cdata,v.untyped_storage()._cdata]);pitches.append(k.stride(0)//d)
        if len(set(owners))!=len(owners):raise ValueError('Compaction KV roots must be unique')
        ptr=np.asarray([[s.sram_k.data_ptr(),s.sram_v.data_ptr(),s.sram_cum.data_ptr(),s.sram_age.data_ptr()] for s in states],dtype=np.int64)
        dst=np.asarray([[s.sram_k.data_ptr(),s.sram_v.data_ptr(),pitch] for s,pitch in zip(states,pitches)],dtype=np.int64)
        arrays=[ptr,np.asarray(indices,dtype=np.int32),np.asarray(pitches,dtype=np.int32),dst]
        descriptors=[torch.from_numpy(a).to(self.hot.device) for a in arrays]
        scratch=torch.empty((2,len(states),h,rows,d),dtype=torch.bfloat16,device=self.hot.device)
        meta=torch.empty((2,len(states),rows),dtype=torch.float32,device=self.hot.device)
        self.descriptor_h2d_bytes+=sum(a.nbytes for a in arrays);self.scratch_peak_bytes=max(self.scratch_peak_bytes,scratch.numel()*2+meta.numel()*4)
        stream=torch.cuda.current_stream(self.hot.device)
        for t in [self.hot,*[t for state in states for t in [state.sram_k,state.sram_v,state.sram_cum,state.sram_age]]]:t.record_stream(stream)
        kernel().compact(descriptors[0],descriptors[1],descriptors[2],scratch,meta,descriptors[3],self.hot,first)
        for i,s in enumerate(states):
            s.sram_k=s.sram_k[:,:rows];s.sram_v=s.sram_v[:,:rows]
            s.sram_cum=self.hot[0,first+i,:rows];s.sram_age=self.hot[1,first+i,:rows]
    @property
    def owned_gpu_bytes(self):return self.hot.numel()*4+self.cold.numel()*4
