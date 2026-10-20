"""Research exact key scan and selected-value reads; no ranking changes."""
import ast
from functools import lru_cache
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import torch
from transformers.models.mistral.modeling_mistral import rotate_half
from gpu_query_read_indices import GpuLiveReader, GpuReadQueryPolicy, READ_AST
from pool_query_selector import PoolQuerySelector
from query_visible_suffix import SuffixQuerySelector, SUFFIX_AST


@lru_cache(maxsize=1)
def kernel():
    from torch.utils.cpp_extension import load
    source=Path(__file__).resolve().parent
    build=Path('/data/nishant/tieredkv_external/channel_gather_v1')
    build.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MAX_JOBS','1')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    return load(name='tieredkv_channel_gather_v1', sources=[str(source/'native_channel_gather.cpp'),
        str(source/'native_channel_gather.cu')], build_directory=str(build),
        extra_cflags=['-O2'],extra_cuda_cflags=['-O3'])


def write_selected_values(layer,fast_rows,cold_key,selected_value,relative):
    key,value=layer.keys,layer.values
    if (relative.ndim!=1 or relative.dtype!=torch.int64
            or key.ndim!=4 or key.shape!=value.shape or key.shape[0]!=1 or cold_key.ndim!=3
            or selected_value.ndim!=3 or cold_key.shape[0]!=key.shape[1]
            or selected_value.shape!=(key.shape[1],len(relative),key.shape[3])
            or cold_key.shape[2]!=key.shape[3]
            or fast_rows<0 or key.shape[2]!=fast_rows+len(relative)
            or relative.device!=key.device
            or any(t.dtype!=key.dtype or t.device!=key.device or t.requires_grad
                   for t in (key,value,cold_key,selected_value))):
        raise ValueError('Selected-value suffix geometry/dtype/device mismatch')
    if any(t.stride(3)!=1 or t.stride(2)<t.shape[3] or t.stride(1)<t.shape[2]*t.stride(2)
           for t in (key,value)):
        raise ValueError('Selected-value suffix requires nonoverlapping row/head strides')
    outputs={t.untyped_storage()._cdata for t in (key,value)}
    sources={t.untyped_storage()._cdata for t in (cold_key,selected_value,relative)}
    if len(outputs)!=2 or outputs&sources:
        raise ValueError('Selected-value suffix ownership overlaps')
    torch.index_select(cold_key,1,relative,out=key[0,:,fast_rows:,:])
    value[0,:,fast_rows:,:].copy_(selected_value)


class ChannelReader(GpuLiveReader):
    def __init__(self,policy):
        super().__init__(policy)
        self.key_reads=self.value_reads=self.key_rows=self.value_rows=0

    def _read(self,layer,channel,relative=None):
        self.indices(layer)
        cache=self.policy.layer_caches[layer];manager=self.manager
        if cache._prefetched is not None:
            raise RuntimeError('Unconsumed ordinary prefetched cold rows')
        manager.flush()
        with torch.cuda.device(manager.device):
            manager.wait()
            self.blocks.record_stream(torch.cuda.current_stream(manager.device))
            source=self.blocks[layer,:,:self.lengths[layer]]
            if relative is not None: source=source.index_select(1,relative)
            source=source.contiguous()
            output=torch.empty((*source.shape,manager.dim),dtype=manager.pool.dtype,device=manager.device)
            count=source.numel()
            if count:
                kernel().gather(manager.pool,source,output,channel)
                manager.h2d_calls+=1;cache.h2d_calls+=1;manager.record()
        cache.host_read_bytes+=count*manager.dim*manager.pool.element_size()
        self.reads+=1
        if channel==0: self.key_reads+=1;self.key_rows+=source.shape[1]
        else: self.value_reads+=1;self.value_rows+=source.shape[1]
        return output.to(cache.cfg.resolved_kv_dtype)

    def read_keys(self,layer): return self._read(layer,0)
    def read_values(self,layer,relative): return self._read(layer,1,relative)

    def close(self):
        super().close()
        self.key_reads=self.value_reads=self.key_rows=self.value_rows=0


class ChannelQueryPolicy(GpuReadQueryPolicy):
    def __init__(self,**options):
        super().__init__(**options)
        self.query_reader=ChannelReader(self)


def channel_factory():
    # Rebuild the proved suffix AST, then alter only key/value delivery.
    tree=ast.parse(textwrap.dedent(inspect.getsource(PoolQuerySelector._hook)))
    select=tree.body[0].body[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='live':
            node.value=ast.parse('self.policy.query_reader.indices(index)',mode='eval').body
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='cache.select_stt_pair':
            node.func=ast.parse('self.policy.query_reader.read_live',mode='eval').body
            node.args=[ast.Name(id='index',ctx=ast.Load())]
    assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==READ_AST['candidate']
    start=next(i for i,n in enumerate(select.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='keys')
    select.body[start:start+5]=ast.parse('''replace_cold_suffix(layer,len(keep),cold_key,cold_value,relative)
self.visible_suffix_updates+=1
''').body
    control=ast.dump(tree,include_attributes=False)
    assert hashlib.sha256(control.encode()).hexdigest()==SUFFIX_AST['candidate']
    changed=[]
    for i,node in enumerate(select.body):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Tuple) and ast.unparse(node.targets[0])=='(cold_key, cold_value)':
            select.body[i]=ast.parse('cold_key=self.policy.query_reader.read_keys(index)').body[0];changed.append('key_read')
        elif isinstance(node,ast.If) and 'cold_key.device' in ast.unparse(node.test):
            node.test=ast.parse('cold_key.device!=query.device',mode='eval').body;changed.append('device_guard')
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='replace_cold_suffix':
            select.body[i]=ast.parse('write_selected_values(layer,len(keep),cold_key,self.policy.query_reader.read_values(index,relative),relative)').body[0]
            changed.append('selected_values')
    assert changed==['key_read','device_guard','selected_values']
    candidate=ast.dump(tree,include_attributes=False)
    namespace=dict(torch=torch,rotate_half=rotate_half,write_selected_values=write_selected_values)
    exec(compile(ast.fix_missing_locations(tree),'<query-channel-reads>','exec'),namespace)
    return namespace['_hook'],dict(control=hashlib.sha256(control.encode()).hexdigest(),
        candidate=hashlib.sha256(candidate.encode()).hexdigest(),changes=changed,ranking_math_unchanged=True)


CHANNEL_HOOK,CHANNEL_AST=channel_factory()


class ChannelQuerySelector(SuffixQuerySelector): _hook=CHANNEL_HOOK
