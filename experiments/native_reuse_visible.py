"""Opt-in gather into consumed raw KV with a verified cached native build."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from reuse_visible_policy import ReuseVisiblePolicy


@lru_cache(maxsize=1)
def kernel():
    source=Path(__file__).resolve().parent
    build=Path('/data/nishant/tieredkv_external/reuse_visible_build_20261002')
    binary=build/'tieredkv_reuse_hot_v1.so'
    proof_path=source.parent/'results/hardware/reuse_visible_integrity_20261002/verified_report.json'
    if proof_path.exists() and binary.exists():
        proof=json.loads(proof_path.read_text())
        if (not proof['verified'] or not proof['candidate_ready_for_opt_in_cli'] or
            hashlib.sha256(binary.read_bytes()).hexdigest()!=proof['reuse_binary_sha256'] or
            any(hashlib.sha256((source/name).read_bytes()).hexdigest()!=proof['native_source_sha256'][f'experiments/{name}']
                for name in ('native_hot_compact.cpp','native_reuse_hot.cu'))):
            raise ValueError('Native reuse binary/sources differ from the verified build')
        if str(torch.__version__)!=proof['torch_version'] or list(sys.version_info[:2])!=proof['python_major_minor']:
            raise ValueError('Verified native reuse requires its measured Torch/Python ABI')
        spec=importlib.util.spec_from_file_location('tieredkv_reuse_hot_v1',binary)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from torch.utils.cpp_extension import load
    build.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    return load(name='tieredkv_reuse_hot_v1',sources=[str(source/'native_hot_compact.cpp'),
        str(source/'native_reuse_hot.cu')],build_directory=str(build),
        extra_cflags=['-O2'],extra_cuda_cflags=['-O3'])


def gather_into(states,destinations):
    """Caller must validate all source/destination bounds and aliases first."""
    first=states[0];heads,_,dim=first.sram_k.shape
    rows=len(first.pending_keep);device=first.sram_k.device
    if device.type!='cuda':raise ValueError('Native reuse requires CUDA tensors')
    pointer_array=np.asarray([[getattr(state,name).data_ptr() for name in
        ('sram_k','sram_v','sram_cum','sram_age')]+[key.data_ptr(),value.data_ptr()]
        for state,(key,value) in zip(states,destinations)],dtype=np.int64)
    size_array=np.asarray([[state.sram_k.shape[1],key.shape[2]]
        for state,(key,_) in zip(states,destinations)],dtype=np.int32)
    keeps=np.asarray([state.pending_keep for state in states],dtype=np.int32)
    pointers=torch.from_numpy(pointer_array).to(device)
    indices=torch.from_numpy(keeps).to(device);sizes=torch.from_numpy(size_array).to(device)
    geometry=torch.empty((2,len(states),heads,0,dim),dtype=torch.bfloat16,device=device)
    metadata=torch.empty((2,len(states),rows),dtype=torch.float32,device=device)
    stream=torch.cuda.current_stream(device)
    for state,pair in zip(states,destinations):
        for tensor in (state.sram_k,state.sram_v,state.sram_cum,state.sram_age,*pair):tensor.record_stream(stream)
    kernel().compact(pointers,indices,sizes,geometry,metadata)
    for i,(state,(key,value)) in enumerate(zip(states,destinations)):
        state.sram_k=key.as_strided((heads,rows,dim),key.stride()[1:])
        state.sram_v=value.as_strided((heads,rows,dim),value.stride()[1:])
        state.sram_cum=metadata.as_strided((rows,),(1,),i*rows)
        state.sram_age=metadata.as_strided((rows,),(1,),(len(states)+i)*rows)


class NativeReusePolicy(ReuseVisiblePolicy):
    hot_gather=staticmethod(gather_into)
