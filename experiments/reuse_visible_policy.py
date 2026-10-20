"""Research-only consumption of decode raw KV buffers; no new native kernel."""
import ast
import inspect
import textwrap
import numpy as np
import torch
from fused_bounded_pool import FusedBoundedPoolPolicy
from fused_hot_compaction import gather
from longbench_eval import TieredKVPolicy


def visible_method():
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVPolicy._visible_tuple)))
    original=ast.dump(tree);replaced=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
        target=node.targets[0]
        if isinstance(target,ast.Name) and target.id in ('vis_k','vis_v'):
            channel='k' if target.id=='vis_k' else 'v'
            assert ast.dump(node.value)==ast.dump(ast.parse(f'torch.cat([fast_{channel}, cold_{channel}], dim=1)',mode='eval').body)
            node.value=ast.parse(f'self._compose_visible(fast_{channel}, cold_{channel}, {channel})',mode='eval').body
            replaced.append(target.id)
    assert sorted(replaced)==['vis_k','vis_v']
    tree=ast.fix_missing_locations(tree);modified=ast.dump(tree)
    env={'torch':torch};exec(compile(tree,'<reuse_production_visibility>','exec'),env)
    return env['_visible_tuple'],original,modified


VISIBLE_REUSE,CONTROL_AST,REUSE_AST=visible_method()


def can_reuse(states,destinations,quota):
    if not states or len(states)!=len(destinations):return False
    geometry=set();inputs=set()
    for state,(key,value) in zip(states,destinations):
        if state.sram_k.ndim!=3 or state.sram_v.shape!=state.sram_k.shape:return False
        heads,_,dim=state.sram_k.shape;rows=len(state.pending_keep)
        keep=np.asarray(state.pending_keep)
        if (keep.ndim!=1 or not np.issubdtype(keep.dtype,np.integer) or
            np.any(keep<0) or np.any(keep>=state.sram_k.shape[1]) or
            state.sram_cum.shape!=state.sram_age.shape or state.sram_cum.shape!=(state.sram_k.shape[1],)):
            return False
        if (key.shape!=value.shape or key.ndim!=4 or key.shape[0]!=1 or
            key.shape[1]!=heads or key.shape[3]!=dim or key.shape[2]<rows+quota or
            any(t.dtype!=torch.bfloat16 or t.device!=state.sram_k.device or not t.is_contiguous()
                for t in (key,value,state.sram_k,state.sram_v)) or
            any(t.dtype!=torch.float32 or not t.is_contiguous() or t.device!=key.device
                for t in (state.sram_cum,state.sram_age))):return False
        geometry.add((heads,dim,rows,state.sram_k.device))
        inputs.update(t.untyped_storage()._cdata for t in
            (state.sram_k,state.sram_v,state.sram_cum,state.sram_age))
        output={key.untyped_storage()._cdata,value.untyped_storage()._cdata}
        if inputs&output or len(output)!=2:return False
    outputs=[t.untyped_storage()._cdata for pair in destinations for t in pair]
    return len(geometry)==1 and len(outputs)==len(set(outputs)) and not inputs&set(outputs)


def gather_into(states,destinations):
    """Inputs must pass can_reuse; write hot prefixes, retain row-strided views."""
    indices=torch.from_numpy(np.asarray([s.pending_keep for s in states],dtype=np.int64)).to(states[0].sram_k.device)
    for state,(key,value),index in zip(states,destinations,indices):
        rows=len(index);heads,_,dim=state.sram_k.shape
        hot_key=key.as_strided((heads,rows,dim),key.stride()[1:])
        hot_value=value.as_strided((heads,rows,dim),value.stride()[1:])
        torch.index_select(state.sram_k,1,index,out=hot_key)
        torch.index_select(state.sram_v,1,index,out=hot_value)
        state.sram_k,state.sram_v=hot_key,hot_value
        state.sram_cum=state.sram_cum.index_select(0,index)
        state.sram_age=state.sram_age.index_select(0,index)


class ReuseVisiblePolicy(FusedBoundedPoolPolicy):
    hot_gather=staticmethod(gather_into)
    def __init__(self,**options):
        super().__init__(**options)
        self.reused_layers=0;self.reuse_fallback_layers=0

    def reset(self):
        super().reset()
        self.reused_layers=0;self.reuse_fallback_layers=0

    def _visible_tuple(self,past):
        groups={}
        for layer,cache in self.layer_caches.items():
            if cache.pending_keep is not None:
                shape=(cache.sram_k.shape[0],cache.sram_k.shape[2],len(cache.pending_keep),cache.sram_k.device)
                groups.setdefault(shape,[]).append((layer,cache))
        for group in groups.values():
            states=[cache for _,cache in group]
            destinations=[past[layer] for layer,_ in group]
            reused=can_reuse(states,destinations,self.stt_expose_quota)
            if reused:
                self.hot_gather(states,destinations);self.reused_layers+=len(states)
            else:
                gather(states,[c.pending_keep for c in states]);self.reuse_fallback_layers+=len(states)
            for cache in states:
                cache.pending_keep=None
                if reused:cache.reused_compactions=getattr(cache,'reused_compactions',0)+1
                else:cache.fused_compactions+=1
        return VISIBLE_REUSE(self,past)

    @staticmethod
    def _compose_visible(fast,cold,raw):
        if (fast.untyped_storage()._cdata==raw.untyped_storage()._cdata and
            fast.data_ptr()==raw.data_ptr() and fast.shape[1]+cold.shape[1]<=raw.shape[2] and
            fast.shape[0]==raw.shape[1] and fast.shape[2]==raw.shape[3] and
            fast.stride()==raw.stride()[1:] and fast.dtype==cold.dtype==raw.dtype):
            result=raw.as_strided((fast.shape[0],fast.shape[1]+cold.shape[1],fast.shape[2]),raw.stride()[1:])
            result.narrow(1,fast.shape[1],cold.shape[1]).copy_(cold)
            return result
        return torch.cat((fast,cold),dim=1)
