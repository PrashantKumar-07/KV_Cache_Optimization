"""Research-only exact host root over the unchanged bounded/reuse policy."""
from direct_shared_pool import DirectSharedPool
from exact_pinned_memory import allocate
from native_reuse_visible import NativeReusePolicy


class ExactSharedPool(DirectSharedPool):
    allocate_pool=staticmethod(allocate)


class ExactPinnedPoolPolicy(NativeReusePolicy):
    direct_pool_cls=ExactSharedPool
