"""Fresh-process exact pool memory with the same root/range/GPU/RSS observer."""
import hashlib
from pathlib import Path
import measure_actual_reuse_memory as actual
import measure_fresh_process_memory as driver
from exact_pinned_pool import ExactSharedPool,ExactPinnedPoolPolicy
from exact_pinned_memory import stats


def main():
    original_roots,original_policy,original_allocate,original_write=driver.PinnedRoots,actual.NativeReusePolicy,ExactSharedPool.allocate_pool,driver.atomic_json
    holder={}
    class CapturedRoots(original_roots):
        def __init__(self):
            super().__init__();holder['tracker']=self
    def allocate(shape,*,dtype,device):
        tensor=original_allocate(shape,dtype=dtype,device=device)
        holder['tracker'].track(tensor)
        return tensor
    def write(path,report):
        report.update(exact_root_memory=True,exact_allocation_stats=stats(),
            exact_memory_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        original_write(path,report)
    driver.PinnedRoots=CapturedRoots;actual.NativeReusePolicy=ExactPinnedPoolPolicy
    ExactSharedPool.allocate_pool=staticmethod(allocate);driver.atomic_json=write
    try:
        actual.main()
        assert stats()['live_allocations']==0,'Exact CPU storage remains owned after reset'
    finally:
        driver.PinnedRoots,actual.NativeReusePolicy,driver.atomic_json=original_roots,original_policy,original_write
        ExactSharedPool.allocate_pool=staticmethod(original_allocate)


if __name__=='__main__':main()
