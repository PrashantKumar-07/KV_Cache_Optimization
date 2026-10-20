"""Opt-in exact pinned allocation for the current native cold controller.

Keep bulk transfers, metadata and selection unchanged. Exact host roots use
the existing allocator's same-CUDA-context lifetime contract.
"""
from bulk_payload_policy import BulkPayloadPool
from cold_boundary_policy import ColdBoundaryPolicy
from exact_pinned_memory import allocate


class ExactBulkPayloadPool(BulkPayloadPool):
    allocate_pool = staticmethod(allocate)


class ExactColdBoundaryPolicy(ColdBoundaryPolicy):
    direct_pool_cls = ExactBulkPayloadPool
