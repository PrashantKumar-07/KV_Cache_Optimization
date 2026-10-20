"""Unified model/policy append in a consumed, uniquely owned visible buffer."""
from functools import partial
import numpy as np
import torch
from raw_input_reuse import RawInputPolicy,RawInputCache,gather_in_place
from bulk_promotion_policy import BulkPromotionPolicy
from reserved_decode_append import reserve_visible,AppendLease


def shared_contract(cache,pair,n_new,budget):
    key,value=pair;hot=len(cache.sram_pos)
    if (n_new!=1 or key.ndim!=4 or key.shape!=value.shape or key.shape[0]!=1
            or key.shape[2]<hot+budget+n_new or cache.sram_k is None or cache.sram_v is None
            or cache.sram_k.shape!=cache.sram_v.shape
            or cache.sram_k.shape!=(key.shape[1],hot,key.shape[3])):return False
    tensors=(key,value,cache.sram_k,cache.sram_v)
    if any(t.dtype!=torch.bfloat16 or t.device!=key.device or t.requires_grad for t in tensors):return False
    if key.untyped_storage()._cdata==value.untyped_storage()._cdata:return False
    for canonical,raw in ((cache.sram_k,key),(cache.sram_v,value)):
        if (canonical.data_ptr()!=raw.data_ptr() or canonical.stride()!=raw.stride()[1:]
                or canonical.untyped_storage()._cdata!=raw.untyped_storage()._cdata
                or raw.stride(3)!=1 or raw.stride(2)!=raw.shape[3]
                or raw.stride(1)<raw.shape[2]*raw.shape[3]):return False
    return True


class SharedTransitionCache(RawInputCache):
    shared_immediate_compactions=0
    def _compact_or_queue(self,victims):
        if self._hot_buffer is None:
            return super()._compact_or_queue(victims)
        if self.pending_keep is not None:
            raise RuntimeError('Unconsumed shared hotcompaction')
        fields=(self.sram_k,self.sram_v,self.sram_cum,self.sram_age)
        if not self._supported_hot_layout(fields):return super()._compact_or_queue(victims)
        # Coldreclaim needs compacted scores now; keep the same ordering.
        mask=np.ones(len(self.sram_pos),dtype=np.bool_);mask[victims]=False
        keep=np.flatnonzero(mask).astype(np.int32)
        self.sram_pos=np.asarray(self.sram_pos,dtype=np.int64)[keep].tolist()
        self.sram_grace=torch.from_numpy(self.sram_grace.numpy()[keep])
        self.pending_keep=keep
        gather_in_place(self)
        self.pending_keep=None
        self.shared_immediate_compactions+=1
        self.reused_compactions=getattr(self,'reused_compactions',0)+1


class SharedTransitionPolicy(RawInputPolicy):
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls=partial(SharedTransitionCache,extra_slots=self.reserved_promotion_budget+self.max_decode_append)
        self.reserved_buffer_allocations=self.shared_hot_prefixes=self.shared_transition_bindings=0
    def _visible_tuple(self,past):return reserve_visible(self,super()._visible_tuple(past))
    def _prepare_decode_reads(self):
        layers=list(self.layer_caches);past=self.incoming_past
        if not layers or layers!=list(range(len(past))):return super()._prepare_decode_reads()
        n_new=max(1,past[0][0].shape[2]-self._vis_len[layers[0]])
        roots=[t.untyped_storage()._cdata for pair in past for t in pair]
        supported=len(set(roots))==len(roots) and all(shared_contract(self.layer_caches[i],past[i],n_new,self.promotion_budget) for i in layers)
        if not supported:return super()._prepare_decode_reads()
        for i in layers:
            self.layer_caches[i]._hot_buffer=past[i]
            self.shared_transition_bindings+=1;self.raw_bound_layers+=1
        try:return BulkPromotionPolicy._prepare_decode_reads(self)
        except BaseException:
            for cache in self.layer_caches.values():cache._hot_buffer=None
            raise
    def reset(self):
        super().reset()
        self.reserved_buffer_allocations=self.shared_hot_prefixes=self.shared_transition_bindings=0


class SharedAppendLease(AppendLease):
    def before(self,model,args,options):
        past=options.get('past_key_values')
        if past is not None and hasattr(past,'layers') and any('update' in layer.__dict__ for layer in past.layers):return
        return super().before(model,args,options)
