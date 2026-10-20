"""Research CPU cold timestamps; GPU visibility ordering remains unchanged."""
import ast
from functools import partial
import hashlib
import inspect
import textwrap
import torch
from torch.nn.utils.rnn import pad_sequence
from bulk_promotion_policy import BulkCache,BulkPromotionPolicy
from longbench_eval import TieredKVPolicy
from tiered_kv_cache import TieredKVCache


def eviction_factory():
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVCache.evict_and_demote)))
    before=ast.dump(tree,include_attributes=False)
    node=next(n for n in tree.body[0].body if isinstance(n,ast.If) and ast.unparse(n.test)=='victims')
    node.body=ast.parse('self._compact_or_queue(victims)').body
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='self.stt_last_access' and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.cat':
            full=node.value.args[0].elts[1]
            assert ast.unparse(full.func)=='torch.full'
            device=next(k for k in full.keywords if k.arg=='device');assert ast.unparse(device.value)=='cfg.device'
            device.value=ast.Constant(value='cpu');changed+=1
    assert changed==1
    namespace={'torch':torch};exec(compile(ast.fix_missing_locations(tree),'<CPU-LRU-deferred-eviction>','exec'),namespace)
    return namespace['evict_and_demote'],dict(control=hashlib.sha256(before.encode()).hexdigest(),candidate=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),timestamp_allocation_changes=changed)


CPU_EVICT,EVICTION_AST=eviction_factory()


class CpuLruCache(BulkCache):
    evict_and_demote=CPU_EVICT
    def initial_bifurcation(self,*args,**kwargs):
        result=super().initial_bifurcation(*args,**kwargs)
        if self.stt_last_access.dtype!=torch.float32:
            raise ValueError('CPU timestamp experiment requires float32 metadata')
        self.timestamp_initial_d2h_bytes=self.stt_last_access.numel()*self.stt_last_access.element_size()
        self.stt_last_access=self.stt_last_access.cpu()
        return result


def visible_indices(cache,live,count,bool_live_gather):
    if count<len(live):
        source=getattr(cache,'_visible_access_gpu',cache.stt_last_access)
        access=source[~cache.stt_shadow] if bool_live_gather else source[live]
        order=torch.argsort(access,descending=True)[:count].tolist()
        return sorted(live[o] for o in order)
    return live


class CpuLruPolicy(BulkPromotionPolicy):
    _visible_cold_indices=staticmethod(visible_indices)
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls=partial(CpuLruCache,extra_slots=self.reserved_promotion_budget+self.max_decode_append)
        self.timestamp_h2d_bytes=self.timestamp_refreshes=0
    def _visible_tuple(self,past):
        caches=list(self.layer_caches.values())
        if not caches:return super()._visible_tuple(past)
        if any(c.stt_last_access.device.type!='cpu' or c.stt_last_access.dtype!=torch.float32 or hasattr(c,'_visible_access_gpu') for c in caches):
            raise RuntimeError('Unexpected CPU timestamp state or unconsumed visibility snapshot')
        packed=pad_sequence([c.stt_last_access for c in caches],batch_first=True)
        device=packed.to(caches[0].cfg.device)
        self.timestamp_h2d_bytes+=packed.numel()*packed.element_size();self.timestamp_refreshes+=1
        try:
            for cache,row in zip(caches,device):cache._visible_access_gpu=row[:len(cache.stt_pos)]
            return super()._visible_tuple(past)
        finally:
            for cache in caches:
                if hasattr(cache,'_visible_access_gpu'):del cache._visible_access_gpu
    def reset(self):
        for cache in self.layer_caches.values():
            if hasattr(cache,'_visible_access_gpu'):del cache._visible_access_gpu
        super().reset()
        self.timestamp_h2d_bytes=self.timestamp_refreshes=0
