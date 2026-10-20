"""Forward ownership restoration, exception and fallback regression proof."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from transformers.cache_utils import DynamicCache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from forward_owner_release import OwnerRelease,ForwardOwnerPolicy,BulkPromotionPolicy


class Policy:
    pass


class Model(torch.nn.Module):
    def __init__(self,policy,fail=False):
        super().__init__();self.config=SimpleNamespace(model_type='mistral')
        self.policy=policy;self.fail=fail;self.saw_released=False

    def forward(self,ids,*,past_key_values,use_cache=True):
        self.saw_released=all(c.sram_k is None and c.sram_v is None for c in self.policy.layer_caches.values())
        for i,layer in enumerate(past_key_values.layers):
            new=torch.zeros_like(layer.keys[:,:,:1])
            past_key_values.update(new,new,i)
            if self.fail:raise RuntimeError('injected partial forward')
        return SimpleNamespace(past_key_values=past_key_values)


def fixture():
    past=DynamicCache();policy=Policy();policy.initialized=True
    policy.promotion_budget=2;policy._vis_len={0:8,1:8};policy.lease_restored=False
    policy.layer_caches={}
    for layer in range(2):
        keys=torch.arange(2*8*4,dtype=torch.int16).reshape(1,2,8,4).view(torch.bfloat16).clone()
        past.update(keys,keys.clone(),layer)
        policy.layer_caches[layer]=SimpleNamespace(sram_pos=list(range(6)),
            sram_k=keys[0,:,:6].clone(),sram_v=keys[0,:,:6].clone(),
            sram_cum=torch.zeros(6),sram_age=torch.zeros(6),_hot_buffer=None)
    return policy,past


@pytest.mark.parametrize('fail',(False,True))
def test_payload_owners_expire_and_views_restore_on_partial_forward_exception(fail):
    policy,past=fixture();model=Model(policy,fail);lease=OwnerRelease(model,policy)
    expected=[(c.sram_k.clone(),c.sram_v.clone()) for c in policy.layer_caches.values()]
    handles=[t.untyped_storage()._weak_ref() for c in policy.layer_caches.values() for t in (c.sram_k,c.sram_v)]
    try:
        if fail:
            with pytest.raises(RuntimeError,match='injected partial forward'):
                model(torch.tensor([[1]]),past_key_values=past,use_cache=True)
        else:model(torch.tensor([[1]]),past_key_values=past,use_cache=True)
        assert model.saw_released and lease.pending is None
        assert lease.released_forwards==lease.restored_forwards==1 and lease.released_layers==2
        assert all(torch.UntypedStorage._expired(h) for h in handles)
        for (layer,c),pair in zip(policy.layer_caches.items(),expected):
            assert torch.equal(c.sram_k.view(torch.int16),pair[0].view(torch.int16))
            assert torch.equal(c.sram_v.view(torch.int16),pair[1].view(torch.int16))
            assert c.sram_k.data_ptr()==past.layers[layer].keys.data_ptr()
        assert policy.lease_restored
    finally:
        lease.close()
        for h in handles:torch.UntypedStorage._free_weak_ref(h)
    assert not model._forward_pre_hooks and not model._forward_hooks


@pytest.mark.parametrize('fault',('prefix','dtype','short','multiappend','alias','uninitialized'))
def test_unsupported_forward_keeps_original_owners(fault):
    policy,past=fixture();model=Model(policy);lease=OwnerRelease(model,policy)
    token=torch.tensor([[1]])
    if fault=='prefix':past.layers[1].keys[0,0,0,0]=42
    elif fault=='dtype':past.layers[1].keys=past.layers[1].keys.float()
    elif fault=='short':policy.promotion_budget=4
    elif fault=='multiappend':token=torch.tensor([[1,2]])
    elif fault=='alias':past.layers[1].values=past.layers[1].keys
    else:policy.initialized=False
    before=[(c.sram_k,c.sram_v) for c in policy.layer_caches.values()]
    try:
        lease.before(model,(token,),dict(past_key_values=past,use_cache=True))
        assert lease.pending is None and lease.released_forwards==0
        assert all(c.sram_k is k and c.sram_v is v for c,(k,v) in zip(policy.layer_caches.values(),before))
    finally:lease.close()


def test_restored_prefix_uses_original_bulk_planner_and_clears_lease_flag(monkeypatch):
    policy,past=fixture();model=Model(policy);lease=OwnerRelease(model,policy)
    try:model(torch.tensor([[1]]),past_key_values=past,use_cache=True)
    finally:lease.close()
    policy.incoming_past=tuple((l.keys,l.values) for l in past.layers)
    policy.raw_bound_layers=0;marker=object()
    monkeypatch.setattr(BulkPromotionPolicy,'_prepare_decode_reads',lambda self:marker)
    assert ForwardOwnerPolicy._prepare_decode_reads(policy) is marker
    assert policy.raw_bound_layers==2 and not policy.lease_restored
    assert all(c._hot_buffer==policy.incoming_past[i] for i,c in policy.layer_caches.items())


def test_row_padded_model_cache_is_released_and_restored():
    policy,past=fixture()
    for layer in past.layers:
        for name in ('keys','values'):
            padded=torch.cat((getattr(layer,name),torch.zeros_like(layer.keys[:,:,:1])),dim=2)
            setattr(layer,name,padded[:,:,:8])
        assert not layer.keys.is_contiguous()
    model=Model(policy);lease=OwnerRelease(model,policy)
    try:
        model(torch.tensor([[1]]),past_key_values=past,use_cache=True)
        assert model.saw_released and lease.released_layers==2
        assert all(c.sram_k.shape==(2,6,4) for c in policy.layer_caches.values())
    finally:lease.close()
