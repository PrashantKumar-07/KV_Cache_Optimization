"""Canonical cache ownership, fallback detach, and original reclaim ordering."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from cpu_phase_policy import CpuPhaseCache,CpuPhasePolicy,RECLAIM,protected_view
from cpu_state import CpuStateArena
from pooled_state import StateArena
from pooled_state_policy import PooledStatePolicy
from tiered_kv_cache import TieredKVCache,TieredConfig


def cache():
    c=CpuPhaseCache.__new__(CpuPhaseCache)
    c.cfg=TieredConfig(num_heads=1,head_dim=2,sram_capacity=3,sttram_capacity=2,device='cpu')
    c.manager=None;c._last_use=None;c.pool=None;c._prefetched=None
    c.sram_pos=[0,10,20];c.stt_pos=[0,10,30,40,50]
    c.sram_grace=torch.zeros(3,dtype=torch.bool)
    c.stt_shadow=torch.tensor([True,True,False,False,False]);c.stt_was_vram_resident=torch.tensor([True,True,False,True,False])
    c.stt_peek_streak=torch.arange(5).float()
    c.sram_cum=torch.tensor([1.,3.,5.]);c.sram_age=torch.ones(3);c.stt_last_access=torch.tensor([1.,2.,0.,1.,2.])
    c._logical=[4,2,5,1,0];c._free=[3,6];c._t=8;c.last_floor_protected=0
    c.attach_arena(StateArena(1,8,7,'cpu'),0)
    arena=CpuStateArena(1,8,7);c.attach_cpu(arena,0,SimpleNamespace())
    return c,arena


def test_canonical_flags_streak_and_positions_follow_native_version():
    c,a=cache();view=c.stt_peek_streak
    c.stt_peek_streak=torch.full((5,),7.)
    assert np.array_equal(a.view('streak',0),np.full(5,7,np.float32)) and c.stt_peek_streak.data_ptr()==view.data_ptr()
    a.append_hot(np.array([[99]],np.int64),np.array([1],np.int32))
    assert c.sram_pos==(0,10,20,99) and c.sram_grace.tolist()==[False]*4
    with pytest.raises(RuntimeError,match='bypassed'):c.sram_pos=[1]
    with pytest.raises(ValueError):c.stt_peek_streak=torch.zeros(4)
    c.close()


def test_detach_preserves_state_and_restores_ordinary_mutable_fields():
    c,a=cache();c.detach_cpu()
    assert c._cpu_arena is None and c.sram_pos==[0,10,20] and c._logical==[4,2,5,1,0]
    c.sram_pos.append(99);c.stt_shadow[0]=False
    assert a.view('hot',0).tolist()==[0,10,20] and bool(a.view('shadow',0)[0])
    c.close()


def test_original_reclaim_selection_commits_exact_cpu_fields_and_fifo_slots():
    c,a=cache();oracle=TieredKVCache(c.cfg)
    for field in ['sram_pos','stt_pos','sram_grace','stt_shadow','stt_was_vram_resident','stt_peek_streak','sram_cum','sram_age','stt_last_access']:
        v=getattr(c,field);setattr(oracle,field,v.clone() if isinstance(v,torch.Tensor) else list(v))
    oracle.sram_k=oracle.sram_v=torch.zeros(1,3,2);oracle.stt_k=oracle.stt_v=torch.zeros(1,5,2)
    oracle._select_hot_victims=lambda:[]
    expected=oracle.evict_and_demote(torch.zeros(1,0,2),torch.zeros(1,0,2),[])
    stamp_root=c.stt_last_access.untyped_storage()._cdata
    dropped,reclaimed=RECLAIM(c,[])
    assert (dropped,reclaimed)==(expected[1],expected[3])==(1,2)
    assert list(c.stt_pos)==oracle.stt_pos and c._logical==(1,0) and c._free==(3,6,4,2,5)
    for field in ['stt_shadow','stt_was_vram_resident','stt_peek_streak','stt_last_access']:assert torch.equal(getattr(c,field),getattr(oracle,field))
    assert c.stt_last_access.untyped_storage()._cdata==stamp_root and c.cpu_reclaims==1
    c.close()


def test_unsupported_append_detaches_before_parent_fallback(monkeypatch):
    c,a=cache();p=CpuPhasePolicy(sram_budget=991,stt_budget=1090)
    p.layer_caches={0:c};p.cpu_arena=a;p._vis_len={0:3};p.incoming_past=[(torch.zeros(1,1,5,2),torch.zeros(1,1,5,2))]
    seen=[]
    monkeypatch.setattr(PooledStatePolicy,'_prepare_decode_reads',lambda self:seen.append(self.layer_caches[0]._cpu_arena))
    before=a.owners['version'].copy();p._prepare_decode_reads()
    assert seen==[None] and p.cpu_arena is None and p.cpu_fallback_steps==1 and np.array_equal(before,a.owners['version'])
    c.close();p.layer_caches={};p.reset()

@pytest.mark.parametrize('positions',[[0,1,2,140],[],[-(1<<63),-(1<<63)+1]])
def test_native_position_mask_preserves_reference_predicate(positions):
    cfg=TieredConfig(sink_size=4,window_size=128)
    c=SimpleNamespace(sram_pos=positions,cfg=cfg)
    assert torch.equal(protected_view(np.asarray(positions,dtype=np.int64),cfg),TieredKVCache._hard_protected_mask(c))
