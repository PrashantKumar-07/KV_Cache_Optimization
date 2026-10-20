"""Experimental exact coordinator with boundary-batched hot compaction."""
import ast
import copy
import inspect
import textwrap
import numpy as np
import torch
from batched_eviction_policy import BatchedEvictionPolicy,BatchedVictimCache
from fused_hot_compaction import gather
from tiered_kv_cache import TieredKVCache


def eviction_method():
    """Replace only the original post-victim compaction block with a hook."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVCache.evict_and_demote)))
    original=ast.dump(tree)
    node=next(n for n in tree.body[0].body if isinstance(n,ast.If) and
        isinstance(n.test,ast.Name) and n.test.id=='victims')
    original_compact=copy.deepcopy(node)
    node.body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),
        attr='_compact_or_queue',ctx=ast.Load()),args=[ast.Name(id='victims',ctx=ast.Load())],keywords=[]))]
    tree=ast.fix_missing_locations(tree)
    modified=ast.dump(tree)
    env={'torch':torch}
    exec(compile(tree,'<deferred_production_eviction>','exec'),env)
    function=env['evict_and_demote']
    compact=ast.FunctionDef(name='compact_now',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='victims')],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id='cfg',ctx=ast.Store())],value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='cfg',ctx=ast.Load())),original_compact],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[compact],type_ignores=[]))
    exec(compile(module,'<production_eager_compaction>','exec'),env)
    return function,env['compact_now'],original,modified


DEFERRED_EVICT,EAGER_COMPACT,CONTROL_AST,DEFERRED_AST=eviction_method()


class FusedVictimCache(BatchedVictimCache):
    evict_and_demote=DEFERRED_EVICT

    def __init__(self,cfg):
        self.pending_keep=None
        self.fused_compactions=self.eager_compactions=0
        super().__init__(cfg)

    def _compact_or_queue(self,victims):
        if self.pending_keep is not None:
            raise RuntimeError('Unconsumed deferred hot compaction')
        fields=(self.sram_k,self.sram_v,self.sram_cum,self.sram_age)
        # Cold reclamation reads compacted scores immediately. Keep it eager.
        supported=(len(self.stt_pos)<=self.cfg.sttram_capacity and
            self.sram_k.device.type=='cuda' and self.sram_k.dtype==torch.bfloat16 and
            self.sram_v.dtype==torch.bfloat16 and self.sram_cum.dtype==torch.float32 and
            self.sram_age.dtype==torch.float32 and all(t.is_contiguous() for t in fields) and
            len(self.sram_pos)-len(victims)==self.cfg.sram_capacity)
        if not supported:
            self.eager_compactions+=1
            return EAGER_COMPACT(self,victims)
        mask=np.ones(len(self.sram_pos),dtype=np.bool_)
        mask[victims]=False
        keep=np.flatnonzero(mask).astype(np.int32)
        self.sram_pos=np.asarray(self.sram_pos,dtype=np.int64)[keep].tolist()
        self.sram_grace=torch.from_numpy(self.sram_grace.numpy()[keep])
        self.pending_keep=keep

    def close(self):
        self.pending_keep=None
        super().close()


class FusedEvictionPolicy(BatchedEvictionPolicy):
    direct_cache_cls=FusedVictimCache

    def _visible_tuple(self,past):
        groups={}
        for cache in self.layer_caches.values():
            if cache.pending_keep is not None:
                shape=(cache.sram_k.shape[0],cache.sram_k.shape[2],len(cache.pending_keep),
                       cache.sram_k.device)
                groups.setdefault(shape,[]).append(cache)
        for caches in groups.values():
            gather(caches,[c.pending_keep for c in caches])
            for cache in caches:
                cache.pending_keep=None
                cache.fused_compactions+=1
        return super()._visible_tuple(past)
