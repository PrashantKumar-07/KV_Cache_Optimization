"""Research layer placement into final shared future-cache owners."""
import importlib.util
import weakref
from pathlib import Path
import torch
from transformers import DynamicCache
from cpu_phase_policy import CpuPhasePolicy

_path=Path(__file__).resolve().parents[1]/'results/hardware/layerwise_prefill_placement_v1/candidate.py'
_spec=importlib.util.spec_from_file_location('frozen_early_placement',_path)
_reference=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_reference)


class EarlySharedPolicy(_reference.EarlyPlacement,CpuPhasePolicy):
    def __init__(self,**options):
        super().__init__(**options)
        self.shared_prefill_layers=0;self.shared_prefill_geometry=None

    def reset(self):
        super().reset()
        self.shared_prefill_layers=0;self.shared_prefill_geometry=None


class EarlySharedCache(DynamicCache):
    def __init__(self,policy):
        super().__init__()
        self.policy_ref=weakref.ref(policy)

    def update(self,key,value,layer_idx,*args,**kwargs):
        policy=self.policy_ref()
        if policy is None:raise RuntimeError('Early shared policy owner expired')
        if policy.initialized:return super().update(key,value,layer_idx,*args,**kwargs)
        if key.ndim!=4 or key.shape!=value.shape or key.shape[0]!=1 or key.dtype!=torch.bfloat16 or value.dtype!=key.dtype or value.device!=key.device or key.requires_grad or value.requires_grad:
            raise ValueError('Early shared placement requires batch-one BF16 inference KV')
        cache=policy.place(layer_idx,key,value)
        hot=len(cache.sram_pos);cold=min(policy.stt_expose_quota,len(cache.stt_pos))
        geometry=(key.shape[1],key.shape[3],hot,cold)
        if policy.shared_prefill_geometry not in (None,geometry):raise ValueError('Early shared geometry differs between layers')
        if cache.sram_k.shape!=(geometry[0],hot,geometry[1]) or cache.sram_v.shape!=cache.sram_k.shape or any(t.dtype!=key.dtype or t.device!=key.device or t.requires_grad for t in (cache.sram_k,cache.sram_v)):
            raise ValueError('Early shared hot payload geometry mismatch')
        policy.shared_prefill_geometry=geometry
        owners=[torch.empty((1,geometry[0],hot+cold+1,geometry[1]),device=key.device,dtype=key.dtype) for _ in range(2)]
        for owner,source in zip(owners,(cache.sram_k,cache.sram_v)):owner[0,:,:hot].copy_(source)
        cache.sram_k,cache.sram_v=(owner[0,:,:hot] for owner in owners)
        # Cold tails are filled by the global visibility handoff. The current
        # attention receives the original full prompt, never these future views.
        super().update(*(owner[:,:,:hot+cold] for owner in owners),layer_idx)
        policy.shared_prefill_layers+=1
        return key,value


class SharedPrefillLease(_reference.PrefillLease):
    def before(self,model,positional,options):
        result=super().before(model,positional,options)
        if result is None:return
        positional,prepared=result
        return positional,{**prepared,'past_key_values':EarlySharedCache(self.policy_ref())}
