"""Research one-row reserved DynamicLayer append and shared visible hot owners."""
from types import MethodType
import weakref
import torch
from transformers.cache_utils import DynamicLayer
from bulk_promotion_policy import BulkPromotionPolicy


def capacity(tensor):
    if (tensor.ndim!=4 or tensor.shape[0]!=1 or tensor.storage_offset()!=0
            or tensor.stride(3)!=1 or tensor.stride(2)!=tensor.shape[3]
            or tensor.shape[3]==0 or tensor.stride(1)%tensor.shape[3]):return 0
    rows=tensor.stride(1)//tensor.shape[3]
    if (rows<tensor.shape[2] or tensor.stride(0)!=tensor.shape[1]*rows*tensor.shape[3]
            or tensor.untyped_storage().nbytes()!=tensor.shape[1]*rows*tensor.shape[3]*tensor.element_size()):return 0
    return rows


def append_reserved(layer,key,value):
    old_key,old_value=layer.keys,layer.values
    if (torch.is_grad_enabled() or not layer.is_initialized or old_key is None or old_value is None
            or key.ndim!=4 or key.shape!=value.shape or key.shape[0]!=1 or key.shape[2]!=1
            or old_key.shape!=old_value.shape or key.shape[1::2]!=old_key.shape[1::2]
            or any(t.dtype!=torch.bfloat16 or t.device!=key.device or t.requires_grad for t in (key,value,old_key,old_value))):return None
    rows=old_key.shape[2];caps=[capacity(t) for t in (old_key,old_value)]
    roots=[t.untyped_storage()._cdata for t in (old_key,old_value)]
    if min(caps)<rows+1 or caps[0]!=caps[1] or len(set(roots))!=2:return None
    if any(t.untyped_storage()._cdata in roots for t in (key,value)):return None
    outputs=[t.as_strided((1,t.shape[1],rows+1,t.shape[3]),t.stride()) for t in (old_key,old_value)]
    outputs[0][:,:,rows:rows+1].copy_(key);outputs[1][:,:,rows:rows+1].copy_(value)
    layer.keys,layer.values=outputs
    return layer.keys,layer.values


class ReservedAppendPolicy(BulkPromotionPolicy):
    def __init__(self,**options):
        super().__init__(**options)
        self.reserved_buffer_allocations=self.shared_hot_prefixes=0
    def _visible_tuple(self,past):
        return reserve_visible(self,super()._visible_tuple(past))
    def reset(self):
        super().reset();self.reserved_buffer_allocations=self.shared_hot_prefixes=0


def reserve_visible(self,result):
        prepared=[]
        for index,(key,value) in enumerate(result):
            cache=self.layer_caches.get(index)
            if (cache is None or key.ndim!=4 or key.shape!=value.shape or key.shape[0]!=1
                    or any(t.dtype!=torch.bfloat16 or t.requires_grad for t in (key,value))
                    or self._vis_sram_keep[index]!=list(range(len(cache.sram_pos)))):
                prepared.append((key,value));continue
            rows=key.shape[2]
            if min(capacity(key),capacity(value))<rows+1:
                buffers=[torch.empty((1,key.shape[1],rows+1,key.shape[3]),device=key.device,dtype=key.dtype) for _ in range(2)]
                for target,source in zip(buffers,(key,value)):target[:,:,:rows].copy_(source)
                key,value=(target[:,:,:rows] for target in buffers)
                self.reserved_buffer_allocations+=1
            hot=len(cache.sram_pos)
            cache.sram_k,cache.sram_v=key[0,:,:hot],value[0,:,:hot]
            self.shared_hot_prefixes+=1;prepared.append((key,value))
        return tuple(prepared)


class AppendLease:
    """Override only owned DynamicLayer instances during a serial model forward."""
    def __init__(self,model,policy):
        self.policy=weakref.ref(policy);self.pending=[];self.fast_updates=self.fallback_updates=0
        self.handles=[]
        try:
            self.handles.append(model.register_forward_pre_hook(self.before,with_kwargs=True))
            self.handles.append(model.register_forward_hook(self.after,with_kwargs=True,always_call=True))
        except BaseException:
            self.close()
            raise
    def before(self,model,args,options):
        if self.pending:raise RuntimeError('Reentrant reserved append lease')
        policy=self.policy();past=options.get('past_key_values');ids=options.get('input_ids',args[0] if args else None)
        if (policy is None or not policy.initialized or ids is None or ids.ndim!=2 or ids.shape[0]!=1
                or not options.get('use_cache',False) or past is None or not hasattr(past,'layers')
                or getattr(model.config,'model_type',None)!='mistral' or any(type(layer) is not DynamicLayer for layer in past.layers)):return
        for layer in past.layers:
            original=layer.update;had_override='update' in layer.__dict__;previous=layer.__dict__.get('update')
            self.pending.append((layer,had_override,previous))
            def update(entry,key,value,*args,original=original,**kwargs):
                result=append_reserved(entry,key,value)
                if result is None:self.fallback_updates+=1;return original(key,value,*args,**kwargs)
                self.fast_updates+=1;return result
            layer.update=MethodType(update,layer)
    def after(self,*args,**kwargs):
        for layer,had_override,previous in self.pending:
            if had_override:layer.update=previous
            else:del layer.update
        self.pending.clear()
    def close(self):
        self.after()
        for handle in self.handles:handle.remove()
        self.handles.clear()
