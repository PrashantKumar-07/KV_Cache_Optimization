"""Research-only prefill hot-prefix backing shared with visible model KV."""
import torch
from forward_owner_release import ForwardOwnerPolicy


def share_prefill_hot(policy,visible):
    bindings=[];raw_ids=[]
    if list(policy.layer_caches)!=list(range(len(visible))):return 0
    for layer,cache in policy.layer_caches.items():
        key,value=visible[layer];hot=len(cache.sram_pos)
        if (key.ndim!=4 or key.shape!=value.shape or key.shape[0]!=1 or
            key.shape[2]<hot or cache.sram_k.shape!=cache.sram_v.shape or
            cache.sram_k.shape!=(key.shape[1],hot,key.shape[3]) or
            any(t.dtype!=torch.bfloat16 or t.device!=key.device
                for t in (key,value,cache.sram_k,cache.sram_v)) or
            any(t.stride(3)!=1 or t.stride(2)!=t.shape[3] or
                t.stride(1)<t.shape[2]*t.shape[3] for t in (key,value))):
            return 0
        if not all(torch.equal(raw[0,:,:hot].view(torch.int16),old.view(torch.int16))
                   for raw,old in ((key,cache.sram_k),(value,cache.sram_v))):return 0
        raw_ids.extend((key.untyped_storage()._cdata,value.untyped_storage()._cdata))
        bindings.append((cache,(key,value),hot))
    metadata={t.untyped_storage()._cdata for c in policy.layer_caches.values()
              for t in (c.sram_cum,c.sram_age)}
    if len(raw_ids)!=len(set(raw_ids)) or metadata.intersection(raw_ids):return 0
    for cache,pair,hot in bindings:
        cache.sram_k,cache.sram_v=(raw[0,:,:hot] for raw in pair)
    return len(bindings)


class PrefillSharedPolicy(ForwardOwnerPolicy):
    prefill_shared_layers=0

    def reset(self):
        self.prefill_shared_layers=0
        super().reset()

    def _visible_tuple(self,past):
        visible=super()._visible_tuple(past)
        if self.decode_step_count==0:
            self.prefill_shared_layers+=share_prefill_hot(self,visible)
        return visible
