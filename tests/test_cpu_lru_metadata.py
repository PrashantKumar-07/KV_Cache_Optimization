"""Cold timestamp allocation and GPU-order-preserving visibility invariants."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from cpu_lru_metadata import visible_indices,CpuLruPolicy,EVICTION_AST
from longbench_eval import TieredKVPolicy


@pytest.mark.parametrize('boolean',[False,True])
@pytest.mark.parametrize('count',[0,1,4,8])
def test_visibility_uses_same_sort_input_and_chosen_rows(boolean,count):
    cache=SimpleNamespace(stt_last_access=torch.tensor([3.,0.,3.,2.,0.,3.,2.,3.]),stt_shadow=torch.tensor([False,True,False,False,False,False,False,False]))
    live=torch.where(~cache.stt_shadow)[0].tolist()
    expected=TieredKVPolicy._visible_cold_indices(cache,live,count,boolean)
    cache._visible_access_gpu=cache.stt_last_access.clone()
    assert visible_indices(cache,live,count,boolean)==expected
    cache.stt_last_access.fill_(99)
    assert visible_indices(cache,live,count,boolean)==expected


def test_only_timestamp_allocation_and_original_fused_compaction_change():
    assert EVICTION_AST['timestamp_allocation_changes']==1


def test_empty_visibility_delegates_and_reset_clears_metadata(monkeypatch):
    from bulk_promotion_policy import BulkPromotionPolicy
    p=CpuLruPolicy(sram_budget=991,stt_budget=1090,start_size=4,recent_size=128)
    monkeypatch.setattr(BulkPromotionPolicy,'_visible_tuple',lambda self,past:past)
    assert p._visible_tuple('empty')=='empty'
    p.timestamp_h2d_bytes=99;p.timestamp_refreshes=9;p.reset()
    assert p.timestamp_h2d_bytes==p.timestamp_refreshes==0


def test_visibility_error_removes_all_temporary_snapshot_views(monkeypatch):
    from bulk_promotion_policy import BulkPromotionPolicy
    p=CpuLruPolicy(sram_budget=991,stt_budget=1090,start_size=4,recent_size=128)
    caches=[SimpleNamespace(stt_last_access=torch.zeros(n),stt_pos=list(range(n)),cfg=SimpleNamespace(device='cpu')) for n in (4,7)]
    p.layer_caches=dict(enumerate(caches))
    def fail(*args):
        assert all(c._visible_access_gpu.device.type=='cpu' for c in caches)
        raise ValueError('injected visibility failure')
    monkeypatch.setattr(BulkPromotionPolicy,'_visible_tuple',fail)
    with pytest.raises(ValueError,match='injected'):p._visible_tuple(None)
    assert all(not hasattr(c,'_visible_access_gpu') for c in caches)
    assert p.timestamp_h2d_bytes==2*7*4 and p.timestamp_refreshes==1
    p.layer_caches={};p.reset()
