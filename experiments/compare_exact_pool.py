"""Same recorded model trial with cached versus exact host roots; no new math."""
import hashlib
from pathlib import Path
import compare_reuse_visible as comparison
import compare_cpu_batched as trial
from check_exact_pool_requests import RecordedExactPolicy
from exact_pinned_memory import stats


def main():
    original_policy,original_candidate,original_write=trial.FusedBoundedPolicy,comparison.RecordedNativePolicy,comparison.atomic_json
    files=list(Path('src').glob('*.py'))+list(Path('experiments').glob('*.py'))+list(Path('experiments').glob('*.cu'))+list(Path('experiments').glob('*.cpp'))
    identity=dict(exact_root_comparison=True,
        arm_mapping={'fused_bounded':'cached_native_reuse','native_reuse':'exact_native_reuse'},
        exact_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    def write(path,report):
        report.update(identity,exact_allocation_stats=stats())
        original_write(path,report)
    trial.FusedBoundedPolicy=original_candidate
    comparison.RecordedNativePolicy=RecordedExactPolicy
    comparison.atomic_json=write
    try:
        comparison.main()
        assert stats()['live_allocations']==0,'Exact root remains owned after model reset'
    finally:
        trial.FusedBoundedPolicy,comparison.RecordedNativePolicy,comparison.atomic_json=original_policy,original_candidate,original_write


if __name__=='__main__':main()
