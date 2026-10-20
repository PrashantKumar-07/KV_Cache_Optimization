"""Ownership and metadata ordering for the research pooled controller."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from pooled_state import StateArena
from pooled_state_policy import PooledCache,PooledStatePolicy,PooledDecode,projection
from longbench_eval import TieredKVPolicy


def cache():
    c=PooledCache.__new__(PooledCache)
    c.sram_cum=torch.tensor([2.,3.]);c.sram_age=torch.tensor([4.,5.]);c.stt_last_access=torch.tensor([7.])
    c._t=8;c.manager=None;c._last_use=None
    c.attach_arena(StateArena(2,6,5,'cpu'),1)
    return c


def test_owner_survives_extensions_and_cold_reorder():
    c=cache();root=c.sram_cum.untyped_storage()._cdata
    c.sram_cum=c.extend_metadata('sram_cum',2)
    assert c.sram_cum.tolist()==[2,3,0,0] and c.sram_cum.untyped_storage()._cdata==root
    c.stt_last_access=c.append_timestamps(2)
    assert c.stt_last_access.tolist()==[7,8,8]
    c.stt_last_access=c.stt_last_access[[2,0]]
    assert c.stt_last_access.tolist()==[8,7] and c.stt_last_access.data_ptr()==c._state_arena.cold[1].data_ptr()


def test_prepared_tail_preserves_promoted_age_and_new_zero():
    c=cache();c._pool_prepared=True
    c._state_arena.hot[1,1,2:5]=torch.tensor([1.,1.,0.])
    c.sram_age=c.extend_metadata('sram_age',2)
    assert c.sram_age.tolist()==[4,5,1,1]
    c.sram_age=c.extend_metadata('sram_age',1)
    assert c.sram_age.tolist()==[4,5,1,1,0]

@pytest.mark.parametrize('fault',['hot','cold','dtype','shape'])
def test_invalid_assignment_or_extension_leaves_logical_state(fault):
    c=cache();before=c.sram_cum.clone()
    with pytest.raises(ValueError):
        if fault=='hot':c.extend_metadata('sram_cum',5)
        elif fault=='cold':c.append_timestamps(5)
        elif fault=='dtype':c.sram_cum=c.sram_cum.double()
        else:c.sram_cum=c.sram_cum.unsqueeze(0)
    assert torch.equal(c.sram_cum,before)


def test_fallback_projection_bumps_age_once_and_new_age_zero():
    c=cache();c.cfg=SimpleNamespace(device='cpu',dtype=torch.float32);c.sram_pos=[0,1]
    cumulative,age=projection(SimpleNamespace(_pool_prepared=False),c,1)
    assert cumulative.tolist()==[2,3,0] and age.tolist()==[5,6,0]
    assert c.sram_age.tolist()==[4,5]


def test_decode_override_is_scoped_and_before_original():
    mro=PooledStatePolicy.__mro__
    assert mro.index(PooledDecode)<mro.index(TieredKVPolicy)
    assert PooledDecode._decode_step is not TieredKVPolicy._decode_step


def test_generated_prepare_helper_cannot_collide_with_original_loop_local():
    from pooled_state_policy import POOLED_PREPARE
    assert 'pooled_projection' in POOLED_PREPARE.__code__.co_names
    assert 'pooled_projection' not in POOLED_PREPARE.__code__.co_varnames
    assert 'projection' in POOLED_PREPARE.__code__.co_varnames
def test_hot_compaction_helper_preserves_ineligible_cpu_pending_payload():
 from types import SimpleNamespace
 import numpy as np
 import torch
 from pooled_state_policy import PooledStatePolicy
 policy=PooledStatePolicy()
 keys=torch.arange(16).reshape(1,4,4).float()
 pending=np.array([0,2],dtype=np.int32)
 cache=SimpleNamespace(sram_k=keys,pending_keep=pending)
 policy.layer_caches={0:cache}
 policy._compact_pending_hot()  # No arena is available yet.
 def unexpected(*args):raise AssertionError('CPU payload must retain original fallback')
 policy.state_arena=SimpleNamespace(compact=unexpected)
 policy._compact_pending_hot();policy._compact_pending_hot()
 assert cache.pending_keep is pending and cache.sram_k is keys
 torch.testing.assert_close(keys,torch.arange(16).reshape(1,4,4).float())
 assert policy.pooled_compact_groups==0

