"""Compose verified bounded slots and exact fused eviction without new math."""
from functools import partial
from bounded_pinned_pool import BoundedCpuCache,BoundedPoolPolicy
from fused_eviction_policy import FusedVictimCache,FusedEvictionPolicy


class FusedBoundedCache(BoundedCpuCache,FusedVictimCache):
    """Bounded constructor wraps the unchanged fused cache constructor."""


class FusedBoundedPoolPolicy(FusedEvictionPolicy,BoundedPoolPolicy):
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls=partial(FusedBoundedCache,
            extra_slots=self.reserved_promotion_budget+self.max_decode_append)
