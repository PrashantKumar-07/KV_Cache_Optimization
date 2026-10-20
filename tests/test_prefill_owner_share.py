"""Prefill byte preservation, released old owners and atomic fallbacks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from prefill_owner_share import share_prefill_hot,PrefillSharedPolicy,ForwardOwnerPolicy


def fixture():
    policy=SimpleNamespace(layer_caches={});visible=[]
    for layer in range(2):
        pair=tuple((torch.arange(2*8*4,dtype=torch.int16)+layer*100).reshape(1,2,8,4).view(torch.bfloat16).clone()
                   for _ in range(2))
        visible.append(pair)
        policy.layer_caches[layer]=SimpleNamespace(sram_pos=list(range(6)),
            sram_k=pair[0][0,:,:6].clone(),sram_v=pair[1][0,:,:6].clone(),
            sram_cum=torch.arange(6,dtype=torch.float32),sram_age=torch.arange(6,dtype=torch.float32))
    return policy,tuple(visible)


def test_prefill_hot_alias_preserves_bits_metadata_and_expires_old_owners():
    policy,visible=fixture()
    expected=[(c.sram_k.clone(),c.sram_v.clone(),c.sram_cum.clone(),c.sram_age.clone()) for c in policy.layer_caches.values()]
    handles=[t.untyped_storage()._weak_ref() for c in policy.layer_caches.values() for t in (c.sram_k,c.sram_v)]
    try:
        assert share_prefill_hot(policy,visible)==2
        assert all(torch.UntypedStorage._expired(h) for h in handles)
        for (layer,c),want in zip(policy.layer_caches.items(),expected):
            for name,value in zip(('sram_k','sram_v','sram_cum','sram_age'),want):
                assert torch.equal(getattr(c,name).contiguous().view(torch.uint8),value.contiguous().view(torch.uint8))
            assert c.sram_k.data_ptr()==visible[layer][0].data_ptr()
    finally:
        for h in handles:torch.UntypedStorage._free_weak_ref(h)


@pytest.mark.parametrize('fault',('prefix','dtype','short','layers','alias','metadata_alias'))
def test_invalid_second_layer_does_not_change_any_hot_owner(fault):
    policy,visible=fixture();visible=list(visible)
    if fault=='prefix':visible[1][0][0,0,0,0]=42
    elif fault=='dtype':visible[1]=tuple(t.float() for t in visible[1])
    elif fault=='short':visible[1]=tuple(t[:,:,:5] for t in visible[1])
    elif fault=='layers':visible.pop()
    elif fault=='alias':visible[1]=visible[0]
    else:policy.layer_caches[1].sram_cum=visible[1][0].view(torch.float32).reshape(-1)[:6]
    before=[(c.sram_k,c.sram_v) for c in policy.layer_caches.values()]
    assert share_prefill_hot(policy,visible)==0
    assert all(c.sram_k is key and c.sram_v is value for c,(key,value) in zip(policy.layer_caches.values(),before))


@pytest.mark.parametrize('step',(0,1))
def test_only_prefill_shares_hot_backing(monkeypatch,step):
    fixture_policy,visible=fixture()
    policy=PrefillSharedPolicy.__new__(PrefillSharedPolicy)
    policy.layer_caches=fixture_policy.layer_caches
    policy.decode_step_count=step;policy.prefill_shared_layers=0
    monkeypatch.setattr(ForwardOwnerPolicy,'_visible_tuple',lambda self,past:past)
    assert policy._visible_tuple(visible) is visible
    assert policy.prefill_shared_layers==(2 if step==0 else 0)
    assert (policy.layer_caches[0].sram_k.data_ptr()==visible[0][0].data_ptr())==(step==0)
