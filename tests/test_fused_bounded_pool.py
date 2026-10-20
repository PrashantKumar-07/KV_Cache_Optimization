"""Verify composition keeps bounded guards ahead of all cache mutations."""
from functools import partial
from types import SimpleNamespace
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from fused_bounded_pool import FusedBoundedCache,FusedBoundedPoolPolicy
from bounded_pinned_pool import BoundedCpuCache,BoundedPoolPolicy
from fused_eviction_policy import FusedVictimCache,FusedEvictionPolicy
from direct_shared_pool import DirectScorePolicy


def prepared():
    policy=FusedBoundedPoolPolicy(sram_budget=991,stt_budget=1090,promotion_budget=32)
    cfg=SimpleNamespace(sram_capacity=991,sttram_capacity=1090)
    policy.layer_caches={i:SimpleNamespace(cfg=cfg,sram_pos=list(range(991)),
        stt_pos=list(range(1090)),capacity=1154,prepared_promotion=None,prepared_victims=None) for i in range(2)}
    policy.manager=SimpleNamespace(geometry=(8,1154))
    policy._vis_len={0:1023,1:1023}
    return policy


def past(appends):
    return tuple((torch.empty(1,1,1023+n,1),None) for n in appends)


@pytest.mark.parametrize('fault',('append','layers','ragged','hot','cold','budget','mutated_append','capacity'))
def test_bad_geometry_rejected_before_prepare_or_decode(fault,monkeypatch):
    policy=prepared();incoming=past((1,1));called=[]
    monkeypatch.setattr(DirectScorePolicy,'_decode_step',lambda *args:called.append(args))
    if fault=='append':incoming=past((33,33))
    elif fault=='layers':policy.layer_caches.pop(1)
    elif fault=='ragged':incoming=past((1,2))
    elif fault=='hot':policy.layer_caches[0].sram_pos.append(991)
    elif fault=='cold':policy.layer_caches[0].stt_pos.append(1090)
    elif fault=='budget':policy.promotion_budget=33
    elif fault=='mutated_append':policy.max_decode_append=64;incoming=past((64,64))
    else:policy.layer_caches[0].capacity+=1
    before=(policy.decode_step_count,tuple((tuple(c.sram_pos),tuple(c.stt_pos)) for c in policy.layer_caches.values()))
    with pytest.raises(ValueError):policy._decode_step(incoming)
    assert not called and policy.incoming_past is None
    assert before==(policy.decode_step_count,tuple((tuple(c.sram_pos),tuple(c.stt_pos)) for c in policy.layer_caches.values()))
    policy.layer_caches={};policy.manager=None


def test_supported_append_and_factory_preserve_both_components(monkeypatch):
    policy=prepared();incoming=past((32,32));marker=object()
    monkeypatch.setattr(DirectScorePolicy,'_decode_step',lambda self,value:marker if value is incoming else None)
    assert policy._decode_step(incoming) is marker
    assert policy.incoming_past is None
    assert isinstance(policy.direct_cache_cls,partial)
    assert policy.direct_cache_cls.func is FusedBoundedCache
    assert policy.direct_cache_cls.keywords=={'extra_slots':64}
    assert FusedBoundedCache.__init__ is BoundedCpuCache.__init__
    assert FusedBoundedCache.evict_and_demote is FusedVictimCache.evict_and_demote
    assert FusedBoundedPoolPolicy._visible_tuple is FusedEvictionPolicy._visible_tuple
    assert BoundedPoolPolicy in FusedBoundedPoolPolicy.__mro__
    policy.layer_caches={};policy.manager=None


@pytest.mark.parametrize('options',({'max_decode_append':0},{'max_decode_append':True},
    {'max_decode_append':1.5},{'sram_budget':32,'promotion_budget':32,'max_decode_append':2},
    {'sram_budget':63,'start_size':4,'recent_size':128}))
def test_constructor_guards_survive_composition(options):
    with pytest.raises(ValueError):FusedBoundedPoolPolicy(**options)
