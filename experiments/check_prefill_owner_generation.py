"""Untimed prefill-owner preservation against ordinary bulk promotion."""
import hashlib
from pathlib import Path
import sys
import check_reuse_integrity as driver
from bulk_promotion_policy import BulkPromotionPolicy
from forward_owner_release import OwnerRelease
from prefill_owner_share import PrefillSharedPolicy


class BulkGeneration(driver.Trajectory, BulkPromotionPolicy):
    def __init__(self, **options):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)


class SharedGeneration(driver.Trajectory, PrefillSharedPolicy):
    counts = {}

    def __init__(self, **options):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)

    def reset(self):
        type(self).counts = dict(prefill_shared_layers=self.prefill_shared_layers,
            raw_bound_layers=self.raw_bound_layers,
            raw_compactions=self.raw_compactions)
        super().reset()


def main():
    if '--native-only' in sys.argv:
        raise ValueError('This driver validates generation, not the old native kernel')
    previous = driver.Control, driver.Candidate, driver.measure, driver.atomic_json

    def measure(model, *args, **kwargs):
        chosen = kwargs['policy_cls']; leases = []; result = None
        before = len(model._forward_pre_hooks), len(model._forward_hooks)
        if chosen is SharedGeneration:
            def factory(**options):
                policy = chosen(**options)
                leases.append(OwnerRelease(model, policy))
                return policy
            kwargs['policy_cls'] = factory
        try:
            result = previous[2](model, *args, **kwargs)
            result['forward_owner_counts'] = dict(
                released_forwards=sum(l.released_forwards for l in leases),
                restored_forwards=sum(l.restored_forwards for l in leases),
                released_layers=sum(l.released_layers for l in leases))
            assert all(l.pending is None and l.released_forwards == l.restored_forwards
                       for l in leases)
            return result
        finally:
            for lease in leases:
                lease.close()
            assert before == (len(model._forward_pre_hooks), len(model._forward_hooks))
            if result is not None:
                result['owner_hooks_removed'] = True

    def write(path, report):
        report.update(actual_control='ordinary_bulk_promotion',
            actual_candidate='research_prefill_owner_share',
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations='Reused development documents, one seed, serialized scoped hooks. '
                'Untimed preservation only; no new native-kernel proof, quality superiority, '
                'serving concurrency, full-resource equality or ordinary adoption.')
        previous[3](path, report)

    driver.Control, driver.Candidate = BulkGeneration, SharedGeneration
    driver.measure, driver.atomic_json = measure, write
    try:
        driver.main()
    finally:
        driver.Control, driver.Candidate, driver.measure, driver.atomic_json = previous


if __name__ == '__main__':
    main()
