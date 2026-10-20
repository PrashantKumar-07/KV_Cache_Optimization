"""Research-only serial Mistral/DynamicLayer ownership release during decode."""
import weakref
import torch
from transformers.cache_utils import DynamicLayer
from raw_input_reuse import RawInputPolicy
from bulk_promotion_policy import BulkPromotionPolicy


class ForwardOwnerPolicy(RawInputPolicy):
    lease_restored = False

    def _prepare_decode_reads(self):
        if not self.lease_restored: return super()._prepare_decode_reads()
        try:
            for layer,cache in self.layer_caches.items():
                pair=self.incoming_past[layer]
                assert all(old.data_ptr()==raw.data_ptr() and old.stride()==raw.stride()[1:]
                           for old,raw in ((cache.sram_k,pair[0]),(cache.sram_v,pair[1])))
                cache._hot_buffer=pair
                self.raw_bound_layers+=1
            return BulkPromotionPolicy._prepare_decode_reads(self)
        finally: self.lease_restored=False

    def reset(self):
        self.lease_restored=False
        super().reset()


class OwnerRelease:
    """Keep only the mutable model cache container while old payloads expire."""
    def __init__(self,model,policy):
        self.policy=weakref.ref(policy)
        self.pending=None
        self.released_forwards=self.restored_forwards=self.released_layers=0
        self.handles=[model.register_forward_pre_hook(self.before,with_kwargs=True),
            model.register_forward_hook(self.after,with_kwargs=True,always_call=True)]

    def before(self,model,args,kwargs):
        if self.pending is not None: raise RuntimeError('Reentrant forward ownership lease')
        policy=self.policy()
        token=kwargs.get('input_ids',args[0] if args else None)
        past=kwargs.get('past_key_values')
        if (policy is None or not policy.initialized or token is None or token.shape!=(1,1) or
            not kwargs.get('use_cache',False) or past is None or not hasattr(past,'layers') or
            getattr(getattr(model,'config',None),'model_type',None)!='mistral'):
            return
        caches=policy.layer_caches
        if list(caches)!=list(range(len(past.layers))) or not caches: return
        raw_ids=[];specs=[]
        for layer,cache in caches.items():
            entry=past.layers[layer]
            if type(entry) is not DynamicLayer: return
            key,value=entry.keys,entry.values
            hot=len(cache.sram_pos)
            if (key is None or value is None or key.ndim!=4 or key.shape!=value.shape or
                key.shape[0]!=1 or key.shape[2]!=policy._vis_len[layer] or
                key.shape[2]<hot+policy.promotion_budget or
                any(t is None or t.dtype!=torch.bfloat16 or t.device!=key.device
                    for t in (key,value,cache.sram_k,cache.sram_v)) or
                any(t.stride(3)!=1 or t.stride(2)!=t.shape[3] or
                    t.stride(1)<t.shape[2]*t.shape[3] for t in (key,value)) or
                cache.sram_k.shape!=cache.sram_v.shape or
                cache.sram_k.shape!=(key.shape[1],hot,key.shape[3])):
                return
            if not all(torch.equal(raw[0,:,:hot].view(torch.int16),old.view(torch.int16))
                       for raw,old in ((key,cache.sram_k),(value,cache.sram_v))): return
            raw_ids.extend((key.untyped_storage()._cdata,value.untyped_storage()._cdata))
            specs.append((layer,hot))
        if len(raw_ids)!=len(set(raw_ids)): return
        metadata={t.untyped_storage()._cdata for cache in caches.values()
                  for t in (cache.sram_cum,cache.sram_age)}
        if metadata.intersection(raw_ids): return
        self.pending=(past,specs)
        policy.lease_restored=False
        for layer,_ in specs:
            caches[layer].sram_k=caches[layer].sram_v=None
        self.released_forwards+=1;self.released_layers+=len(specs)

    def after(self,model,args,kwargs,out):
        if self.pending is None: return
        policy=self.policy();past,specs=self.pending
        try:
            if policy is None: return
            for layer,hot in specs:
                entry=past.layers[layer];cache=policy.layer_caches[layer]
                cache.sram_k=entry.keys[0,:,:hot]
                cache.sram_v=entry.values[0,:,:hot]
            self.restored_forwards+=1
            policy.lease_restored=True
            if out is not None and out.past_key_values is not past:
                raise RuntimeError('Forward ownership lease requires the same DynamicCache')
        finally: self.pending=None

    def close(self):
        if self.pending is not None: self.after(None,(),{},None)
        for handle in self.handles: handle.remove()
        self.handles=[]
