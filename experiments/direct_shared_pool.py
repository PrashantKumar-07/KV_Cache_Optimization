"""Opt-in direct shared-pool initialization; existing decode transfer machinery."""
import torch
from cpu_batched_cache import SharedPool, CpuBatchCache, CpuBatchPolicy
from cpu_slot_cache import transfer_kernels
from batched_score_policy import batched_update
from tiered_kv_cache import TieredKVCache


class DirectSharedPool(SharedPool):
    @staticmethod
    def allocate_pool(shape, *, dtype, device):
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    def __init__(self, first, layers):
        self.caches = {}
        self.d2h, self.h2d = transfer_kernels()
        self.pending = []; self.pending_slots = set(); self.last_use = None
        self.d2h_calls = self.h2d_calls = 0
        self.device, self.dim = first.cfg.device, first.cfg.head_dim
        self.span = first.cfg.num_heads * first.capacity
        self.geometry = (first.cfg.num_heads, first.capacity, first.cfg.resolved_stt_kv_dtype, str(first.cfg.device))
        self.pool = self.allocate_pool((2,layers*self.span,1,self.dim),
            dtype=first.cfg.resolved_stt_kv_dtype,device=self.device)

    def attach(self, cache):
        geometry = (cache.cfg.num_heads,cache.capacity,cache.cfg.resolved_stt_kv_dtype,str(cache.cfg.device))
        if geometry != self.geometry or cache.cfg.head_dim != self.dim:
            raise ValueError('Direct shared initialization requires homogeneous layer geometry')
        number = len(self.caches)
        if (number+1)*self.span > self.pool.shape[1]:
            raise ValueError('Direct shared pool layer capacity exceeded')
        self.caches[number] = cache
        cache.base = number*self.span
        cache.pool = self.pool[:,cache.base:cache.base+self.span]
        cache.manager = self


class DirectCpuCache(CpuBatchCache):
    def initial_bifurcation(self,*args,**kwargs):
        if self.manager is None or self.pool is None or self._logical:
            raise ValueError('Direct cache must be freshly attached before initialization')
        view = self.pool
        self.pool = None  # Core placement writes the temporary GPU cold tensors.
        try:
            counts = TieredKVCache.initial_bifurcation(self,*args,**kwargs)
        finally:
            self.pool = view
        count = len(self.stt_pos)
        if count > self.capacity:
            raise ValueError('Initial cold rows exceed slot capacity')
        self._logical = list(range(count)); self._free = list(range(count,self.capacity))
        self.manager.enqueue(self,self._raw_k,self._raw_v,self._logical)
        # Flush each layer: no accumulated full-model cold GPU staging allocation.
        self.manager.flush()
        self.host_initial_write_bytes = self.host_write_bytes
        self._raw_k = self._raw_v = None
        return counts


class DirectScorePolicy(CpuBatchPolicy):
    update_scores = batched_update
    direct_cache_cls = DirectCpuCache
    direct_pool_cls = DirectSharedPool

    def _bifurcate(self,past, **prefill_state):
        original = self.cache_cls
        def create(cfg):
            cache = self.direct_cache_cls(cfg)
            if self.manager is None:
                self.manager = self.direct_pool_cls(cache,len(past))
            self.manager.attach(cache)
            return cache
        self.cache_cls = create
        try:
            return super()._bifurcate(past, **prefill_state)
        finally:
            self.cache_cls = original
