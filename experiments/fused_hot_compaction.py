"""Experimental BF16/F32 homogeneous cross-layer hot-state gather."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch


@lru_cache(maxsize=1)
def kernel():
    source = Path(__file__).resolve().parent
    build = Path('/data/nishant/tieredkv_external/fused_hot_compact_build_20261002')
    binary = build/'tieredkv_hot_compact_v1.so'
    evidence = source.parent/'results/hardware/fused_hot_compaction_screen_20261002'
    proof_path = evidence/'verified_report_v2.json'
    if proof_path.exists() and binary.exists():
        proof = json.loads(proof_path.read_text())
        hashes = json.loads((evidence/'source_v2_manifest.json').read_text())['source_sha256']
        runtime = json.loads((evidence/'result_v2.json').read_text())['runtime']
        if (not proof['verified'] or not proof['advance_to_model'] or
            hashlib.sha256(binary.read_bytes()).hexdigest()!=proof['binary_sha256'] or
            any(hashlib.sha256((source/name).read_bytes()).hexdigest()!=hashes[f'experiments/{name}']
                for name in ('native_hot_compact.cpp','native_hot_compact.cu'))):
            raise ValueError('Native hot-compaction binary/sources differ from the verified build')
        if str(torch.__version__)!=runtime['packages']['torch'] or sys.version_info[:2]!=(3,10):
            raise ValueError('Verified hot-compaction binary requires its measured Torch/Python ABI')
        spec = importlib.util.spec_from_file_location('tieredkv_hot_compact_v1',binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from torch.utils.cpp_extension import load
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '1')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    return load(name='tieredkv_hot_compact_v1', sources=[str(source/'native_hot_compact.cpp'),
        str(source/'native_hot_compact.cu')], build_directory=str(build),
        extra_cflags=['-O2'], extra_cuda_cflags=['-O3'])


def gather(states, keeps):
    """Allocate and gather all KV/score tensors; preserve every payload bit."""
    if not states or len(states)!=len(keeps):
        raise ValueError('Expected a nonempty complete gather plan')
    first = states[0]
    heads,_,dim = first.sram_k.shape
    rows = len(keeps[0])
    device = first.sram_k.device
    for state,keep in zip(states,keeps):
        tensors = (state.sram_k,state.sram_v,state.sram_cum,state.sram_age)
        old_rows = state.sram_k.shape[1]
        if (state.sram_k.shape!=(heads,old_rows,dim) or state.sram_v.shape!=state.sram_k.shape or
            state.sram_cum.shape!=(old_rows,) or state.sram_age.shape!=(old_rows,) or
            len(keep)!=rows or
            any(t.device!=device or not t.is_contiguous() for t in tensors) or
            state.sram_k.dtype!=torch.bfloat16 or state.sram_v.dtype!=torch.bfloat16 or
            state.sram_cum.dtype!=torch.float32 or state.sram_age.dtype!=torch.float32):
            raise ValueError('Gather requires homogeneous contiguous BF16 KV/F32 scores')
    keep_array = np.asarray(keeps,dtype=np.int32).reshape(len(states),rows)
    size_array = np.asarray([state.sram_k.shape[1] for state in states],dtype=np.int32)
    if np.any(keep_array<0) or np.any(keep_array>=size_array[:,None]):
        raise ValueError('Gather indices exceed source rows')
    pointer_array = np.asarray([[getattr(state,name).data_ptr() for name in
        ('sram_k','sram_v','sram_cum','sram_age')] for state in states],dtype=np.int64)
    pointers = torch.from_numpy(pointer_array).to(device)
    indices = torch.from_numpy(keep_array).to(device)
    sizes = torch.from_numpy(size_array).to(device)
    kv = torch.empty((2,len(states),heads,rows,dim),dtype=torch.bfloat16,device=device)
    metadata = torch.empty((2,len(states),rows),dtype=torch.float32,device=device)
    stream = torch.cuda.current_stream(device)
    for state in states:
        for name in ('sram_k','sram_v','sram_cum','sram_age'):
            getattr(state,name).record_stream(stream)
    kernel().compact(pointers,indices,sizes,kv,metadata)
    for index,state in enumerate(states):
        state.sram_k,state.sram_v = kv[0,index],kv[1,index]
        state.sram_cum,state.sram_age = metadata[0,index],metadata[1,index]


def compact_group(states,victims):
    """Charge CPU masks/positions, descriptors/uploads, allocation and CUDA gather."""
    outputs = [SimpleNamespace(**vars(state)) for state in states]
    active,keeps = [],[]
    for state,selection in zip(outputs,victims):
        if not selection:
            continue
        mask = np.ones(len(state.sram_pos),dtype=np.bool_)
        mask[selection] = False
        keep = np.flatnonzero(mask).astype(np.int32)
        state.sram_pos = np.asarray(state.sram_pos,dtype=np.int64)[keep].tolist()
        state.sram_grace = torch.from_numpy(state.sram_grace.numpy()[keep])
        active.append(state)
        keeps.append(keep)
    if active:
        gather(active,keeps)
    return outputs
