"""Serial ownership and quarantine proof; CUDA integration is a separate gate."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/name) for name in ('experiments','src')]
import reusable_exact_root as module
from cold_boundary_policy import ColdBoundaryPolicy


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(module,'context_key',lambda device:(1,0,99))
    monkeypatch.setattr(module,'allocate',lambda shape,**options:torch.zeros(shape,dtype=options['dtype']))
    return module.ExactRootOwner()

def acquire(owner,shape=(2,4,1,2)):
    return owner.acquire(shape,dtype=torch.bfloat16,device='cuda:0')

def test_one_root_reused_across_serial_leases_and_stale_tokens_reject(owner):
    first=acquire(owner);token=owner.token;first.fill_(7)
    with pytest.raises(RuntimeError,match='already leased'):acquire(owner)
    with pytest.raises(RuntimeError,match='Foreign'):owner.release(object())
    owner.release(token)
    second=acquire(owner)
    assert second is first and torch.all(second==7) and owner.leases==2
    with pytest.raises(RuntimeError,match='stale'):owner.release(token)
    with pytest.raises(RuntimeError,match='Reset'):owner.close()
    owner.release(owner.token);owner.close();owner.close()
    assert owner.root is None
    with pytest.raises(RuntimeError,match='closed'):acquire(owner)

def test_geometry_and_context_changes_reject_without_replacing_idle_root(owner,monkeypatch):
    root=acquire(owner);owner.release(owner.token)
    with pytest.raises(ValueError,match='geometry'):acquire(owner,(2,5,1,2))
    monkeypatch.setattr(module,'context_key',lambda device:(2,0,99))
    with pytest.raises(ValueError,match='context'):acquire(owner)
    assert owner.root is root and owner.token is None and owner.leases==1

@pytest.mark.parametrize('shape,dtype',[((),torch.bfloat16),((0,),torch.bfloat16),
    ((True,),torch.bfloat16),((1,),torch.float32)])
def test_invalid_request_never_reaches_cuda(shape,dtype,monkeypatch):
    monkeypatch.setattr(module,'context_key',lambda device:pytest.fail('Invalid input reached CUDA'))
    with pytest.raises(ValueError,match='BF16 geometry'):
        module.ExactRootOwner().acquire(shape,dtype=dtype,device='cuda:0')

def test_policy_releases_only_after_complete_reset_and_quarantines_failure(owner,monkeypatch):
    root=acquire(owner);token=owner.token
    policy=module.ReusableExactPolicy(root_owner=owner)
    manager=module.ReusableBulkPool.__new__(module.ReusableBulkPool)
    manager.root_owner=owner
    manager.root_token=token
    cache=SimpleNamespace(pool=root,manager=manager)
    manager.caches={0:cache}
    policy.manager=manager;policy.layer_caches={0:cache}
    def fail(policy):raise RuntimeError('injected reset failure')
    monkeypatch.setattr(ColdBoundaryPolicy,'reset',fail)
    with pytest.raises(RuntimeError,match='injected'):policy.reset()
    assert owner.token is token and cache.pool is root
    with pytest.raises(RuntimeError,match='already leased'):acquire(owner)
    def complete(policy):
        assert owner.token is token
        cache.pool=None;cache.manager=None
        policy.manager=None;policy.layer_caches={}
    monkeypatch.setattr(ColdBoundaryPolicy,'reset',complete)
    policy.reset()
    assert owner.token is None and owner.root is root

def test_reset_without_owned_manager_does_not_release_another_policy_lease(owner,monkeypatch):
    acquire(owner);token=owner.token
    policy=module.ReusableExactPolicy(root_owner=owner)
    monkeypatch.setattr(ColdBoundaryPolicy,'reset',lambda policy:None)
    policy.reset()
    assert owner.token is token

def test_failed_initialization_detaches_manager_cache_before_releasing_lease(owner,monkeypatch):
    root=acquire(owner);token=owner.token
    policy=module.ReusableExactPolicy(root_owner=owner)
    manager=module.ReusableBulkPool.__new__(module.ReusableBulkPool)
    manager.root_owner=owner
    manager.root_token=token
    cache=SimpleNamespace(pool=root,manager=manager)
    def close():
        assert owner.token is token
        cache.pool=None;cache.manager=None
    cache.close=close;manager.caches={0:cache}
    policy.manager=manager;policy.layer_caches={}  # Attached before bifurcation failed.
    def complete(policy):
        cache.manager=None;manager.caches.clear();policy.manager=None
    monkeypatch.setattr(ColdBoundaryPolicy,'reset',complete)
    policy.reset()
    assert cache.pool is None and owner.token is None

def test_foreign_manager_rejects_before_parent_reset(owner,monkeypatch):
    other=module.ExactRootOwner();acquire(other);token=other.token
    policy=module.ReusableExactPolicy(root_owner=owner)
    manager=module.ReusableBulkPool.__new__(module.ReusableBulkPool)
    manager.root_owner=other;manager.root_token=token;manager.caches={}
    policy.manager=manager
    monkeypatch.setattr(ColdBoundaryPolicy,'reset',lambda policy:pytest.fail('Touched foreign manager'))
    with pytest.raises(RuntimeError,match='Foreign'):policy.reset()
    assert other.token is token

def test_constructor_failure_detaches_root_and_never_releases_foreign_lease(owner,monkeypatch):
    failed=[]
    def initialize(pool,*args):
        pool.pool=pool.allocate_pool((2,4,1,2),dtype=torch.bfloat16,device='cuda:0')
        failed.append(pool)
        raise RuntimeError('injected pool construction failure')
    monkeypatch.setattr(module.BulkPayloadPool,'__init__',initialize)
    with pytest.raises(RuntimeError,match='injected'):
        module.ReusableBulkPool(root_owner=owner)
    assert failed[0].pool is None and owner.token is None
    acquire(owner);token=owner.token
    with pytest.raises(RuntimeError,match='already leased'):
        module.ReusableBulkPool(root_owner=owner)
    assert owner.token is token

def test_cold_acquire_on_foreign_thread_rejects_before_allocation(owner):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(acquire,owner)
        with pytest.raises(ValueError,match='changed thread'):future.result()
        future=executor.submit(owner.close)
        with pytest.raises(ValueError,match='changed thread'):future.result()
    assert owner.root is None and owner.token is None

def test_reentrant_acquire_release_during_provisioning_rejects(owner,monkeypatch):
    def provision(shape,**options):
        with pytest.raises(RuntimeError,match='already leased'):acquire(owner)
        with pytest.raises(RuntimeError,match='incomplete'):owner.release(owner.token)
        return torch.zeros(shape,dtype=options['dtype'])
    monkeypatch.setattr(module,'allocate',provision)
    acquire(owner)
    assert owner.token is not None and owner.leases==1

def test_failed_allocation_can_retry_without_leaking_busy_state(owner,monkeypatch):
    def fail(*args,**kwargs):raise MemoryError('injected allocation failure')
    monkeypatch.setattr(module,'allocate',fail)
    with pytest.raises(MemoryError):acquire(owner)
    assert owner.root is None and owner.token is None and not owner.acquiring
    monkeypatch.setattr(module,'allocate',lambda shape,**kw:torch.zeros(shape,dtype=kw['dtype']))
    acquire(owner)
    assert owner.leases==1

def test_stale_manager_token_rejects_before_parent_cleanup(owner,monkeypatch):
    acquire(owner);token=owner.token
    policy=module.ReusableExactPolicy(root_owner=owner)
    manager=module.ReusableBulkPool.__new__(module.ReusableBulkPool)
    manager.root_owner=owner;manager.root_token=object();manager.caches={}
    policy.manager=manager
    monkeypatch.setattr(ColdBoundaryPolicy,'reset',lambda policy:pytest.fail('Touched stale manager'))
    with pytest.raises(RuntimeError,match='Stale'):policy.reset()
    assert owner.token is token
