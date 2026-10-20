"""Exact host roots preserve the default factory and inherited policy guards."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import exact_pinned_memory as memory
from direct_shared_pool import DirectSharedPool,DirectScorePolicy
from exact_pinned_pool import ExactSharedPool,ExactPinnedPoolPolicy
from native_reuse_visible import NativeReusePolicy
from longbench_eval import TieredKVPolicy

def test_exact_cold_pool_preserves_bulk_controller_and_only_replaces_allocation():
    from bulk_payload_policy import BulkPayloadPool
    from cold_boundary_policy import ColdBoundaryPolicy
    from exact_cold_boundary_policy import ExactBulkPayloadPool, ExactColdBoundaryPolicy
    assert ExactColdBoundaryPolicy.direct_pool_cls is ExactBulkPayloadPool
    assert ColdBoundaryPolicy.direct_pool_cls is BulkPayloadPool
    assert ExactBulkPayloadPool.allocate_pool is memory.allocate
    for method in ('__init__', 'queue_bulk', 'flush', 'enqueue', 'read'):
        assert getattr(ExactBulkPayloadPool, method) is getattr(BulkPayloadPool, method)
    for method in ('_decode_step', '_visible_tuple', '_prepare_decode_reads', 'reset'):
        assert getattr(ExactColdBoundaryPolicy, method) is getattr(ColdBoundaryPolicy, method)


@pytest.mark.parametrize('shape,dtype,device',[
    ((),torch.bfloat16,'cuda'),((0,),torch.bfloat16,'cuda'),((-1,),torch.bfloat16,'cuda'),
    ((True,),torch.bfloat16,'cuda'),((1.5,),torch.bfloat16,'cuda'),
    ((1,),torch.int64,'cuda'),((1<<64,),torch.bfloat16,'cuda'),((1,),torch.bfloat16,'cpu')])
def test_invalid_buffer_requests_rejected_before_cuda(shape,dtype,device,monkeypatch):
    monkeypatch.setattr(memory,'driver',lambda:pytest.fail('Invalid input reached the driver'))
    with pytest.raises(ValueError):memory.allocate(shape,dtype=dtype,device=device)


def test_default_factory_keeps_original_torch_pinned_allocation(monkeypatch):
    calls=[];marker=object()
    def empty(shape,**options):calls.append((shape,options));return marker
    monkeypatch.setattr(torch,'empty',empty)
    assert DirectSharedPool.allocate_pool((2,8,1,128),dtype=torch.bfloat16,device='cuda') is marker
    assert calls==[((2,8,1,128),dict(dtype=torch.bfloat16,pin_memory=True))]
    assert DirectScorePolicy.direct_pool_cls is DirectSharedPool
    assert ExactSharedPool.allocate_pool is memory.allocate
    assert ExactPinnedPoolPolicy._decode_step is NativeReusePolicy._decode_step
    assert ExactPinnedPoolPolicy._visible_tuple is NativeReusePolicy._visible_tuple


@pytest.mark.parametrize('fails',(False,True))
def test_pool_factory_selected_once_and_cache_factory_restored(monkeypatch,fails):
    policy=ExactPinnedPoolPolicy(sram_budget=991,stt_budget=1090)
    created=[];attached=[]
    class Pool:
        def __init__(self,cache,layers):created.append((cache,layers))
        def attach(self,cache):attached.append(cache)
    policy.direct_pool_cls=Pool
    policy.direct_cache_cls=lambda cfg:SimpleNamespace(cfg=cfg)
    original=policy.cache_cls
    def placement(self,past):
        for cfg in ('first','second'):self.cache_cls(cfg)
        if fails:raise RuntimeError('placement failed')
        return 'placed'
    monkeypatch.setattr(TieredKVPolicy,'_bifurcate',placement)
    if fails:
        with pytest.raises(RuntimeError,match='placement failed'):policy._bifurcate([None,None])
    else:assert policy._bifurcate([None,None])=='placed'
    assert policy.cache_cls is original
    assert len(created)==1 and created[0][1]==2
    assert [c.cfg for c in attached]==['first','second']
    policy.manager=None


def test_interleaved_counter_hook_reaches_recorded_candidate(monkeypatch,tmp_path):
    import check_exact_pool_requests as check
    original=check.requests.native.NativeReusePolicy
    marker=lambda *args:None
    def run():
        # The inner driver installs its counter hook on this class.
        assert check.requests.native.NativeReusePolicy is original
        monkeypatch.setattr(original,'hot_gather',staticmethod(marker))
        assert check.requests.RecordedNativePolicy.hot_gather is marker
        assert check.requests.RecordedNativePolicy.direct_pool_cls is ExactSharedPool
    monkeypatch.setattr(check.requests,'main',run)
    monkeypatch.setattr(check,'stats',lambda:dict(live_allocations=0))
    monkeypatch.setattr(sys,'argv',['check','--kind','interleaved','--output',str(tmp_path/'proof.json')])
    check.main()
    assert check.requests.native.NativeReusePolicy is original
