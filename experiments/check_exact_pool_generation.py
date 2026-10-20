"""Reuse frozen generation checks with cached versus exact pinned roots."""
import hashlib
from pathlib import Path
import check_reuse_integrity as checks
from exact_pinned_pool import ExactSharedPool
from exact_pinned_memory import stats


class CachedControl(checks.Trajectory, checks.native.NativeReusePolicy):
    def __init__(self, **options):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)


class ExactCandidate(checks.Candidate):
    direct_pool_cls = ExactSharedPool


def main():
    original = checks.Control, checks.Candidate, checks.atomic_json
    files = list(Path('experiments').glob('*.py'))
    identity = dict(
        arm_mapping={'fused_bounded': 'cached_native_reuse',
                     'native_reuse': 'exact_native_reuse'},
        exact_generation_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in files})

    def write(path, report):
        report.update(identity, exact_allocation_stats=stats())
        original[2](path, report)

    checks.Control, checks.Candidate, checks.atomic_json = CachedControl, ExactCandidate, write
    try:
        checks.main()
        assert stats()['live_allocations'] == 0, 'Exact root remains owned after reset'
    finally:
        checks.Control, checks.Candidate, checks.atomic_json = original


if __name__ == '__main__':
    main()
