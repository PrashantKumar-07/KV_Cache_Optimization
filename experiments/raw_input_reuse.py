"""Research-only hot payload reuse of validated incoming model KV."""
import ast
from functools import partial
import hashlib
import inspect
import textwrap
import torch
from bulk_promotion_policy import BulkCache, BulkPromotionPolicy
from batched_eviction_policy import BatchedVictimCache
from fused_eviction_policy import FusedVictimCache
from tiered_kv_cache import TieredKVCache


def payload_method(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    before = ast.dump(tree, include_attributes=False)
    body = tree.body[0].body
    found = [(i,n) for i,n in enumerate(body) if isinstance(n,ast.Assign) and
             isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr in ('sram_k','sram_v')]
    assert len(found) == 2
    (first,key),(second,value) = found
    assert second == first+1
    for node,channel in ((key,'k'),(value,'v')):
        assert ast.unparse(node.value.func) == 'torch.cat'
        assert ast.unparse(node.value.args[0].elts[0]) == 'self.sram_'+channel
        assert ast.unparse(node.value.keywords[0].value) == '1'
    call = ast.Expr(ast.Call(ast.Attribute(ast.Name('self',ast.Load()),'_extend_hot_pair',ast.Load()),
        [key.value.args[0].elts[1],value.value.args[0].elts[1]],[]))
    body[first:second+1] = [call]
    after = ast.dump(tree, include_attributes=False)
    namespace = method.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree),'<raw-hot-payload-only>', 'exec'),namespace)
    body[first:first+1] = [key,value]
    assert ast.dump(tree,include_attributes=False) == before
    return namespace[method.__name__],dict(control=hashlib.sha256(before.encode()).hexdigest(),
        candidate=hashlib.sha256(after.encode()).hexdigest())


def queue_method():
    method = FusedVictimCache._compact_or_queue
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    before = ast.dump(tree,include_attributes=False)
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
             ast.unparse(n) == 'all((t.is_contiguous() for t in fields))']
    assert len(calls) == 1
    node = calls[0]; original = node.func,node.args
    node.func = ast.Attribute(ast.Name('self',ast.Load()),'_supported_hot_layout',ast.Load())
    node.args = [ast.Name('fields',ast.Load())]
    after = ast.dump(tree,include_attributes=False)
    namespace = method.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree),'<raw-hot-layout-only>','exec'),namespace)
    node.func,node.args = original
    assert ast.dump(tree,include_attributes=False) == before
    return namespace[method.__name__],dict(control=hashlib.sha256(before.encode()).hexdigest(),
        candidate=hashlib.sha256(after.encode()).hexdigest())


PROMOTE, PROMOTE_AST = payload_method(TieredKVCache.promote)
APPEND, APPEND_AST = payload_method(TieredKVCache.append_tokens)
QUEUE, QUEUE_AST = queue_method()


def binding_contract(cache, pair, n_new, budget):
    key,value = pair
    hot = len(cache.sram_pos)
    if n_new != 1 or key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
        return False
    if key.shape[2] < hot+budget+n_new or cache.sram_k.shape != cache.sram_v.shape:
        return False
    if tuple(key.shape[1::2]) != (cache.sram_k.shape[0],cache.sram_k.shape[2]):
        return False
    tensors = (key,value,cache.sram_k,cache.sram_v)
    if any(t.dtype != torch.bfloat16 or t.device != key.device for t in tensors): return False
    if not key.is_contiguous() or not value.is_contiguous(): return False
    if key.untyped_storage()._cdata == value.untyped_storage()._cdata: return False
    return all(torch.equal(raw[0,:,:hot].view(torch.int16),old.view(torch.int16))
               for raw,old in ((key,cache.sram_k),(value,cache.sram_v)))


def gather_in_place(cache):
    """Gather each full channel before overwriting it; scratch stays per layer."""
    key,value = cache._hot_buffer
    assert key.untyped_storage()._cdata != value.untyped_storage()._cdata
    assert cache.sram_k.shape == cache.sram_v.shape
    assert cache.sram_cum.shape == cache.sram_age.shape == (cache.sram_k.shape[1],)
    index = torch.as_tensor(cache.pending_keep,dtype=torch.long,device=key.device)
    cum = cache.sram_cum.index_select(0,index)
    age = cache.sram_age.index_select(0,index)
    rows = len(index)
    for name,raw in (('sram_k',key),('sram_v',value)):
        selected = getattr(cache,name).index_select(1,index)
        target = raw[0,:,:rows]
        target.copy_(selected)
        setattr(cache,name,target)
        del selected
    cache.sram_cum,cache.sram_age = cum,age


class RawInputCache(BulkCache):
    _hot_buffer = None
    _compact_or_queue = QUEUE
    append_tokens = APPEND

    def promote(self,hits):
        if self.prepared_promotion is not None:
            return BatchedVictimCache.promote(self,hits)
        return PROMOTE(self,hits)

    def _extend_hot_pair(self,keys,values):
        buffer = self._hot_buffer
        old = self.sram_k.shape[1]; rows = keys.shape[1]
        supported = (buffer is not None and keys.shape == values.shape and
            keys.dtype == values.dtype == self.sram_k.dtype and
            keys.device == values.device == self.sram_k.device and
            old+rows <= buffer[0].shape[2] and
            all(t.untyped_storage()._cdata not in {b.untyped_storage()._cdata for b in buffer}
                for t in (keys,values)))
        if not supported:
            self._hot_buffer = None
            self.sram_k = torch.cat((self.sram_k,keys),dim=1)
            self.sram_v = torch.cat((self.sram_v,values),dim=1)
            return
        for name,source,raw in (('sram_k',keys,buffer[0]),('sram_v',values,buffer[1])):
            raw[0,:,old:old+rows].copy_(source)
            setattr(self,name,raw[0,:,:old+rows])

    @staticmethod
    def _supported_hot_layout(fields):
        key,value,cum,age = fields
        return (key.ndim == 3 and key.shape == value.shape and
            all(t.stride(2) == 1 and t.stride(1) == t.shape[2] and
                t.stride(0) >= t.shape[1]*t.shape[2] for t in (key,value)) and
            cum.is_contiguous() and age.is_contiguous())

    def close(self):
        self._hot_buffer = None
        super().close()


class RawInputPolicy(BulkPromotionPolicy):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.direct_cache_cls = partial(RawInputCache,
            extra_slots=self.reserved_promotion_budget+self.max_decode_append)
        self.raw_bound_layers = self.raw_compactions = 0

    def reset(self):
        super().reset()
        self.raw_bound_layers = self.raw_compactions = 0

    def _prepare_decode_reads(self):
        past = self.incoming_past
        first = next(iter(self.layer_caches))
        n_new = max(1,past[0][0].shape[2]-self._vis_len[first])
        raw_ids = [t.untyped_storage()._cdata for pair in past for t in pair]
        old_ids = {t.untyped_storage()._cdata for c in self.layer_caches.values()
                   for t in (c.sram_k,c.sram_v,c.sram_cum,c.sram_age)}
        if len(raw_ids) == len(set(raw_ids)) and not old_ids.intersection(raw_ids):
            bindings = [(c,past[layer]) for layer,c in self.layer_caches.items()
                        if binding_contract(c,past[layer],n_new,self.promotion_budget)]
            # Validate every prefix before any layer can mutate an incoming buffer.
            for cache,pair in bindings:
                cache._hot_buffer = pair
                cache.sram_k,cache.sram_v = (raw[0,:,:len(cache.sram_pos)] for raw in pair)
                self.raw_bound_layers += 1
        return super()._prepare_decode_reads()

    def _visible_tuple(self,past):
        try:
            for cache in self.layer_caches.values():
                if cache.pending_keep is not None and cache._hot_buffer is not None:
                    gather_in_place(cache)
                    cache.pending_keep = None
                    cache.reused_compactions = getattr(cache,'reused_compactions',0)+1
                    self.raw_compactions += 1; self.reused_layers += 1
            return super()._visible_tuple(past)
        finally:
            for cache in self.layer_caches.values(): cache._hot_buffer = None
