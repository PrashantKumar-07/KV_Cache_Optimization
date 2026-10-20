"""Integrate native plans without global patches; preserve unsupported fallback."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from native_controller_policy import NativeControllerPolicy,ControllerCache,address_tensor
from bulk_promotion_stage import fixture,run
from bulk_promotion_policy import BulkPrepare


def test_crosslayer_native_plans_floor_rows_and_streaks_match_original():
    frame=dict(budget=4,attention={0:{2:.4,3:.7,4:.7},1:{0:1.,3:.2}},attention_indices={0:{2:.4,3:.7,4:.7},1:{0:1.,3:.2}},layers={i:dict(positions=list(range(7)),inclusive=True,floor=3,hysteresis=2,stt_shadow=torch.tensor([False,True,False,False,False,False,False]),stt_was_vram_resident=torch.tensor([True,False,True,False,True,False,False]),stt_peek_streak=torch.arange(7).float()%2) for i in range(2)})
    control,caches=fixture(frame,True);expected,selections,_=run(control,caches,True)
    p=NativeControllerPolicy(sram_budget=991,stt_budget=1090,promotion_budget=4)
    _,p.layer_caches=fixture(frame,True);p._stt_attn=frame['attention'];p._stt_attn_idx=frame['attention_indices'];seen=[];p._prefetch=lambda value:seen.append(value)
    assert p.native_prepare() and p._planned_promotions==expected and seen==[selections]
    for layer,c in p.layer_caches.items():assert torch.equal(c.stt_peek_streak,caches[layer].stt_peek_streak)
    p.layer_caches={};p.manager=None;p.reset()

@pytest.mark.parametrize('fault',['nan','negative','dtype'])
def test_native_prepare_falls_back_before_any_mutation(fault):
    p=NativeControllerPolicy(sram_budget=991,stt_budget=1090)
    c=SimpleNamespace(stt_pos=[0,1],stt_shadow=torch.zeros(2,dtype=torch.bool),stt_was_vram_resident=torch.zeros(2,dtype=torch.bool),stt_peek_streak=torch.zeros(2),cfg=SimpleNamespace(inclusive=True,stt_live_floor=0),_HYSTERESIS_MIN_STREAK=2)
    p.layer_caches={0:c};p._stt_attn={0:{1:1.}};p._stt_attn_idx={0:{(-1 if fault=='negative' else 1):(float('nan') if fault=='nan' else 1.)}}
    if fault=='dtype':c.stt_peek_streak=c.stt_peek_streak.double()
    before=c.stt_peek_streak.clone();assert not p.native_prepare() and torch.equal(before,c.stt_peek_streak) and p.controller_plan_calls==0
    p.layer_caches={};p.manager=None;p.reset()


def test_address_tensor_matches_original_multi_layer_logical_order():
    caches={0:SimpleNamespace(cfg=SimpleNamespace(num_heads=2),capacity=8,base=0,_logical=[6,4,2]),1:SimpleNamespace(cfg=SimpleNamespace(num_heads=1),capacity=8,base=16,_logical=[2,1,7])}
    m=SimpleNamespace(caches=caches,device='cpu',controller_address_calls=0,controller_address_array_peak=0)
    selections={1:[2,0],0:[1,0]}
    expected=[caches[l].base+h*caches[l].capacity+caches[l]._logical[i] for l,idx in selections.items() for h in range(caches[l].cfg.num_heads) for i in idx]
    assert address_tensor(m,selections).tolist()==expected and m.controller_address_calls==1


def test_cached_promotion_rows_validate_before_consume():
    c=ControllerCache.__new__(ControllerCache);c.controller_rows=([(2,3)],[2],4)
    with pytest.raises(RuntimeError,match='identities'):c.promotion_rows([(3,4)])
    assert c.controller_rows is not None and c.promotion_rows([(2,3)])==([2],4) and c.controller_rows is None
    c.manager=None;c._last_use=None


def test_mro_uses_native_prepare_without_global_mutation():
    from native_controller_policy import ControllerPrepare
    before=BulkPrepare._prepare_decode_reads
    assert NativeControllerPolicy.__mro__.index(ControllerPrepare)<NativeControllerPolicy.__mro__.index(BulkPrepare)
    assert BulkPrepare._prepare_decode_reads is before
