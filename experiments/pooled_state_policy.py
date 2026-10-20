"""Research integration of fixed metadata owners and fused state transitions."""
import ast
import inspect
import textwrap
from functools import partial
import torch
from tiered_kv_cache import TieredKVCache
from longbench_eval import TieredKVPolicy
from native_controller_policy import NativeControllerPolicy, ControllerCache, NATIVE_EVICT
from batched_eviction_policy import BatchedEvictionPolicy, BatchedVictimCache
from shared_decode_transition import shared_contract
from bounded_deferred_ownership import gather_contract
from pooled_state import StateArena


def compile_method(method, transform, extra=None):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    transform(tree)
    namespace = method.__globals__.copy()
    namespace.update(extra or {})
    exec(compile(ast.fix_missing_locations(tree), '<pooled-state-controller>', 'exec'), namespace)
    return namespace[method.__name__]


def payload_transform(tree):
    body = tree.body[0].body
    pairs = [(i,n) for i,n in enumerate(body) if isinstance(n,ast.Assign)
             and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr in ('sram_k','sram_v')]
    assert len(pairs) == 2 and pairs[1][0] == pairs[0][0]+1
    first,key = pairs[0]; _,value = pairs[1]
    assert all(ast.unparse(n.value.func) == 'torch.cat' for _,n in pairs)
    body[first:first+2] = [ast.Expr(ast.Call(ast.Attribute(ast.Name('self',ast.Load()),'_extend_hot_pair',ast.Load()),
        [key.value.args[0].elts[1],value.value.args[0].elts[1]],[]))]
    fields = [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute)
              and n.targets[0].attr in ('sram_cum','sram_age')]
    assert len(fields) == 2
    for n in fields:
        assert ast.unparse(n.value.func) == 'torch.cat'
        count = n.value.args[0].elts[1].args[0]
        n.value = ast.Call(ast.Attribute(ast.Name('self',ast.Load()),'extend_metadata',ast.Load()),
                           [ast.Constant(n.targets[0].attr),count],[])

POOLED_PROMOTE = compile_method(TieredKVCache.promote,payload_transform)
POOLED_APPEND = compile_method(TieredKVCache.append_tokens,payload_transform)


def metadata_property(name,channel):
    def get(self): return self.__dict__['_'+name]
    def set(self,value):
        arena = getattr(self,'_state_arena',None)
        if arena is None:
            self.__dict__['_'+name] = value
            return
        owner = arena.cold[self._state_layer] if channel is None else arena.hot[channel,self._state_layer]
        if (value.ndim != 1 or value.dtype != owner.dtype or value.device != owner.device
                or value.requires_grad or len(value)>len(owner)):
            raise ValueError('Unsupported pooled metadata assignment')
        target = owner[:len(value)]
        if value.data_ptr() != target.data_ptr(): target.copy_(value)
        self.__dict__['_'+name] = target
    return property(get,set)


class PooledCache(ControllerCache):
    _state_arena = None
    _pool_prepared = False
    sram_cum = metadata_property('sram_cum',0)
    sram_age = metadata_property('sram_age',1)
    stt_last_access = metadata_property('stt_last_access',None)
    append_tokens = POOLED_APPEND

    def attach_arena(self,arena,layer):
        values = arena.attach(layer,self.sram_cum,self.sram_age,self.stt_last_access)
        self._state_arena,self._state_layer = arena,layer
        self.sram_cum,self.sram_age,self.stt_last_access = values

    def extend_metadata(self,name,count):
        old = getattr(self,name)
        arena = self._state_arena
        if arena is None:
            return torch.cat((old,torch.zeros(count,device=old.device,dtype=old.dtype)))
        owner = arena.hot[0 if name=='sram_cum' else 1,self._state_layer]
        if len(old)+count > len(owner): raise ValueError('Hot metadata exceeds arena')
        if not self._pool_prepared: owner[len(old):len(old)+count].zero_()
        return owner[:len(old)+count]

    def append_timestamps(self,count):
        old = self.stt_last_access
        if self._state_arena is None:
            return torch.cat((old,torch.full((count,),float(self._t),device=old.device,dtype=old.dtype)))
        owner = self._state_arena.cold[self._state_layer]
        if len(old)+count>len(owner): raise ValueError('Cold metadata exceeds arena')
        owner[len(old):len(old)+count].fill_(self._t)
        return owner[:len(old)+count]

    def promote(self,hits):
        if self.prepared_promotion is not None: return BatchedVictimCache.promote(self,hits)
        return POOLED_PROMOTE(self,hits)

    def close(self):
        super().close()
        self._state_arena = None
        self._pool_prepared = False


def eviction_transform(tree):
    body = tree.body[0].body
    compact = next(n for n in body if isinstance(n,ast.If) and ast.unparse(n.test)=='victims')
    compact.body = ast.parse('self._compact_or_queue(victims)').body
    start = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='backup_by_pos')
    end = next(i for i,n in enumerate(body[start:],start) if isinstance(n,ast.If) and ast.unparse(n.test)=='restored_backups')
    body[start:end] = ast.parse('no_backup_victims,restored_backups=self.native_backups(victims)\ndemoted=len(victims)\nwrites_saved=len(restored_backups)').body
    nodes = [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self.stt_last_access'
             and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.cat']
    assert len(nodes)==1
    nodes[0].value = ast.parse('self.append_timestamps(len(no_backup_victims))',mode='eval').body

POOLED_EVICT = compile_method(TieredKVCache.evict_and_demote,eviction_transform)

def pooled_evict(self,*args,**options):
    if not self.cfg.batch_backup_updates: return ControllerCache.evict_and_demote(self,*args,**options)
    if self.stt_shadow.device.type!='cpu' or self.stt_shadow.dtype!=torch.bool or not self.stt_shadow.is_contiguous():
        return ControllerCache.evict_and_demote(self,*args,**options)
    return POOLED_EVICT(self,*args,**options)
PooledCache.evict_and_demote = pooled_evict


def projection(self,cache,n_new):
    if self._pool_prepared:
        n = len(cache.sram_pos)+n_new
        return self.state_arena.hot[0,cache._state_layer,:n],self.state_arena.hot[1,cache._state_layer,:n]
    zeros = torch.zeros(n_new,device=cache.cfg.device,dtype=cache.cfg.dtype)
    return torch.cat((cache.sram_cum,zeros)),torch.cat((cache.sram_age+1,zeros))


def prepare_transform(tree):
    body = tree.body[0].body
    assert ast.unparse(body[0])=='super()._prepare_decode_reads()'
    body[0] = ast.parse('super(BatchedEvictionPolicy,self)._prepare_decode_reads()').body[0]
    loop = next(n for n in body if isinstance(n,ast.For) and ast.unparse(n.target)=='(layer, cache)')
    zero = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='zeros')
    loop.body[zero] = ast.parse('cumulative,age=pooled_projection(self,cache,n_new)').body[0]
    for n in ast.walk(tree):
        if isinstance(n,ast.keyword) and n.arg in ('sram_cum','sram_age'):
            n.value = ast.Name('cumulative' if n.arg=='sram_cum' else 'age',ast.Load())
POOLED_PREPARE = compile_method(BatchedEvictionPolicy._prepare_decode_reads,prepare_transform,
                               dict(BatchedEvictionPolicy=BatchedEvictionPolicy,pooled_projection=projection))


def decode_transform(tree):
    class AgeGuard(ast.NodeTransformer):
        def visit_AugAssign(self,node):
            if ast.unparse(node.target)=='cache.sram_age':
                return ast.If(ast.parse('not self._pool_prepared',mode='eval').body,[node],[])
            return node
    AgeGuard().visit(tree)

class PooledDecode(TieredKVPolicy):
    _decode_step = compile_method(TieredKVPolicy._decode_step,decode_transform)


class PooledStatePolicy(NativeControllerPolicy,PooledDecode):
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls = partial(PooledCache,extra_slots=self.reserved_promotion_budget+self.max_decode_append)
        self.state_arena = None
        self._pool_prepared = False
        self.pooled_prepare_calls = self.pooled_compact_groups = 0

    def native_prepare(self):
        result = super().native_prepare()
        if not result:
            self._pool_prepared = False
            for c in self.layer_caches.values(): c._pool_prepared = False
            return False
        if self._pool_prepared:
            counts = [[len(c.sram_pos),len(c.controller_rows[1]) if c.controller_rows is not None else 0,1]
                      for c in self.layer_caches.values()]
            self.state_arena.prepare(counts)
            self.pooled_prepare_calls += 1
            for c in self.layer_caches.values(): c._pool_prepared = True
        return True

    def _prepare_decode_reads(self):
        self._pool_prepared = False
        for c in self.layer_caches.values(): c._pool_prepared = False
        layers = list(self.layer_caches); past = self.incoming_past
        supported = (self.state_arena is not None and layers == list(range(len(past))) and layers
            and all(c.cfg.write_aware_lambda==0 for c in self.layer_caches.values()))
        if supported:
            n_new = max(1,past[0][0].shape[2]-self._vis_len[layers[0]])
            roots = [t.untyped_storage()._cdata for pair in past for t in pair]
            supported = len(roots)==len(set(roots)) and all(shared_contract(self.layer_caches[i],past[i],n_new,self.promotion_budget) for i in layers)
        if not supported: return super()._prepare_decode_reads()
        for i in layers:
            self.layer_caches[i]._hot_buffer = past[i]
            self.shared_transition_bindings += 1; self.raw_bound_layers += 1
        self._pool_prepared = True
        return POOLED_PREPARE(self)

    def _visible_tuple(self,past):
        if self.state_arena is None and self.layer_caches:
            layers = list(self.layer_caches)
            supported = layers==list(range(len(past))) and all(c.sram_cum.dtype==torch.float32 and c.sram_cum.device.type=='cuda' for c in self.layer_caches.values())
            if supported:
                self.state_arena = StateArena(len(layers),self.sram_budget+self.reserved_promotion_budget+self.max_decode_append,
                    self.stt_budget+self.reserved_promotion_budget+self.max_decode_append,next(iter(self.layer_caches.values())).sram_cum.device)
                for i,c in self.layer_caches.items(): c.attach_arena(self.state_arena,i)
        self._compact_pending_hot()
        return super()._visible_tuple(past)

    def _compact_pending_hot(self):
        """Consume eligible deferred hot compactions without fetching cold KV."""
        if self.state_arena is not None:
            group=[]; first=None; geometry=None
            def flush():
                if not group: return
                self.state_arena.compact(group,[c.pending_keep for c in group],first)
                self.pooled_compact_groups += 1
                for c in group:
                    c.pending_keep=None;c.batched_deferred_compactions+=1
                    c.reused_compactions=getattr(c,'reused_compactions',0)+1
            for i,c in self.layer_caches.items():
                eligible = c.sram_k.device.type=='cuda' and gather_contract(c)
                shape = (c.sram_k.shape[0],c.sram_k.shape[2],len(c.pending_keep)) if eligible else None
                if group and (not eligible or i!=first+len(group) or shape!=geometry or len(group)==8):
                    flush();group=[]
                if eligible:
                    if not group: first=i;geometry=shape
                    group.append(c)
            flush()

    def reset(self):
        super().reset()
        self.state_arena = None;self._pool_prepared = False
        self.pooled_prepare_calls = self.pooled_compact_groups = 0
