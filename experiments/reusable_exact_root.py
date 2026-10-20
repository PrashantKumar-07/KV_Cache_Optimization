"""Opt-in single-context, serial-request exact pinned-root reuse.

Provisioning is a cold-start cost. The owner retains and counts its buffer
between requests; a lease ends only after complete policy reset, including
cache-view cleanup. No concurrent requests or geometry migration is supported.
"""
import ctypes
from functools import partial
import threading
import time
import torch
from exact_pinned_memory import allocate, checked, driver
from bulk_payload_policy import BulkPayloadPool
from cold_boundary_policy import ColdBoundaryPolicy


def context_key(device):
    target=torch.device(device)
    if target.type!='cuda':
        raise ValueError('Pinned root requires a CUDA owner context')
    with torch.cuda.device(target):
        torch.cuda.current_stream(target)
        context=ctypes.c_void_p()
        checked(driver().cuCtxGetCurrent,ctypes.byref(context))
        if not context.value:
            raise RuntimeError('CUDA owner context is unavailable')
        return threading.get_ident(),torch.cuda.current_device(),context.value


class ExactRootOwner:
    def __init__(self):
        self.root=None; self.geometry=None; self.token=None; self.closed=False
        self.allocation_ms=None; self.leases=0
        self.thread=threading.get_ident()
        self.acquiring=False

    def acquire(self,shape,*,dtype,device):
        if self.closed or self.token is not None:
            raise RuntimeError('Exact root owner is closed or already leased')
        if threading.get_ident()!=self.thread:
            raise ValueError('Exact root acquisition changed thread')
        if not shape or any(type(n) is not int or n<1 for n in shape) or dtype!=torch.bfloat16:
            raise ValueError('Exact controller root requires positive BF16 geometry')
        self.token=object()  # Reserve before CUDA calls can release the GIL.
        self.acquiring=True
        try:
            geometry=tuple(shape),dtype,context_key(device)
            if self.geometry is not None and geometry!=self.geometry:
                raise ValueError('Exact root geometry/thread/context changed')
            if self.root is None:
                start=time.perf_counter()
                root=allocate(tuple(shape),dtype=dtype,device=device)
                self.allocation_ms=(time.perf_counter()-start)*1000
                self.root=root;self.geometry=geometry
        except BaseException:
            self.token=None
            raise
        finally:
            self.acquiring=False
        self.leases+=1
        return self.root

    def release(self,token):
        if token is None or token is not self.token:
            raise RuntimeError('Foreign or stale exact-root lease')
        if self.acquiring or self.geometry is None or threading.get_ident()!=self.thread:
            raise RuntimeError('Exact root acquisition is incomplete or on another thread')
        if context_key(f'cuda:{self.geometry[2][1]}')!=self.geometry[2]:
            raise ValueError('Exact root release changed thread/context')
        self.token=None

    def close(self):
        if self.token is not None:
            raise RuntimeError('Reset the active policy before closing its root owner')
        if threading.get_ident()!=self.thread:
            raise ValueError('Exact root close changed thread')
        if self.root is not None and context_key(f'cuda:{self.geometry[2][1]}')!=self.geometry[2]:
            raise ValueError('Exact root close changed thread/context')
        self.root=None;self.closed=True


class ReusableBulkPool(BulkPayloadPool):
    def __init__(self,*args,root_owner):
        self.root_owner=root_owner;self.root_token=None
        def provision(shape,**options):
            root=root_owner.acquire(shape,**options)
            self.root_token=root_owner.token
            return root
        self.allocate_pool=provision
        try:
            super().__init__(*args)
        except BaseException:
            if self.root_token is not None:
                self.pool=None
                root_owner.release(self.root_token)
            raise


class ReusableExactPolicy(ColdBoundaryPolicy):
    def __init__(self,*,root_owner,**options):
        self.root_owner=root_owner
        super().__init__(**options)
        self.direct_pool_cls=partial(ReusableBulkPool,root_owner=root_owner)

    def reset(self):
        manager=self.manager
        if isinstance(manager,ReusableBulkPool) and manager.root_owner is not self.root_owner:
            raise RuntimeError('Foreign exact-root manager')
        if isinstance(manager,ReusableBulkPool) and manager.root_token is not self.root_owner.token:
            raise RuntimeError('Stale exact-root manager lease')
        token=manager.root_token if isinstance(manager,ReusableBulkPool) else None
        caches={id(c):c for c in self.layer_caches.values()}
        if isinstance(manager,ReusableBulkPool):
            caches.update({id(c):c for c in manager.caches.values()})
        super().reset()
        if token is not None:
            # An attached cache can fail bifurcation before layer_caches owns
            # it. Manager close waits IO; explicitly detach that leftover view.
            for cache in caches.values():
                if cache.pool is not None or cache.manager is not None:
                    cache.close()
            if any(c.pool is not None or c.manager is not None for c in caches.values()):
                raise RuntimeError('Exact root still has an outgoing cache owner')
            self.root_owner.release(token)
