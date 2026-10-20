"""Untimed ordinary-bulk versus frozen ownership prototype in the fresh caller."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments'))
import audit_bulk_resources as audit
from bulk_promotion_policy import BulkPromotionPolicy
from forward_owner_release import OwnerRelease
from prefill_owner_share import PrefillSharedPolicy


def counts(policy):
    caches = policy.layer_caches.values()
    result = {name: sum(getattr(c, name, 0) for c in caches) for name in
        ('host_read_bytes', 'host_write_bytes', 'host_initial_write_bytes')}
    result.update(prefill_shared_layers=getattr(policy, 'prefill_shared_layers', 0),
        raw_bound_layers=getattr(policy, 'raw_bound_layers', 0),
        raw_compactions=getattr(policy, 'raw_compactions', 0),
        native_h2d_calls=getattr(policy.manager, 'h2d_calls', 0),
        native_d2h_calls=getattr(policy.manager, 'd2h_calls', 0))
    return result


class CapturedBulkPolicy(BulkPromotionPolicy):
    last_counts = {}

    def reset(self):
        type(self).last_counts = counts(self)
        super().reset()


class CapturedSharedPolicy(PrefillSharedPolicy):
    last_counts = {}

    def reset(self):
        type(self).last_counts = counts(self)
        super().reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('ordinary', 'shared'), required=True)
    parser.add_argument('--hot', type=int, default=991)
    parser.add_argument('--cold', type=int, default=1090)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = audit.fresh.AutoModelForCausalLM, audit.BulkPromotionPolicy, audit.fresh.atomic_json, sys.argv
    models = []; leases = []; baseline = []

    class Loader:
        @staticmethod
        def from_pretrained(*positional, **options):
            model = original[0].from_pretrained(*positional, **options)
            models.append(model)
            baseline.append((len(model._forward_pre_hooks), len(model._forward_hooks)))
            return model

    def factory(**options):
        if args.arm == 'ordinary':
            return CapturedBulkPolicy(**options)
        policy = CapturedSharedPolicy(**options)
        leases.append(OwnerRelease(models[0], policy))
        return policy

    def write(path, report):
        for lease in leases:
            assert lease.pending is None and lease.released_forwards == lease.restored_forwards
            lease.close()
        # audit_bulk_resources retains its one primitive diagnostic hook pair until return.
        if models:
            assert (len(models[0]._forward_pre_hooks), len(models[0]._forward_hooks)) == (
                baseline[0][0] + 1, baseline[0][1] + 1)
        report.update(fresh_owner_arm=args.arm, fresh_owner_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            prototype_counts=(CapturedSharedPolicy if args.arm == 'shared' else CapturedBulkPolicy).last_counts.copy(),
            owner_counts=dict(released_forwards=sum(l.released_forwards for l in leases),
                restored_forwards=sum(l.restored_forwards for l in leases),
                released_layers=sum(l.released_layers for l in leases)), owner_hooks_removed=True,
            fresh_owner_limits='Same frozen fresh caller and standard host allocator. Ordinary BulkPromotionPolicy '
                'versus frozen PrefillSharedPolicy/scoped OwnerRelease. No prefill cleanup, exact-root or VMM. '
                'Untimed resource/preservation audit; no full-resource equality, quality superiority or ordinary adoption.')
        original[2](path, report)

    audit.fresh.AutoModelForCausalLM = Loader
    audit.BulkPromotionPolicy = factory
    audit.fresh.atomic_json = write
    sys.argv = ['audit', '--kind', 'fresh', '--arm', 'bulk', '--hot', str(args.hot),
        '--cold', str(args.cold), '--output', str(args.output)]
    try:
        audit.main()
    finally:
        for lease in leases:
            lease.close()
        if models:
            assert baseline[0] == (len(models[0]._forward_pre_hooks), len(models[0]._forward_hooks))
        audit.fresh.AutoModelForCausalLM, audit.BulkPromotionPolicy, audit.fresh.atomic_json, sys.argv = original


if __name__ == '__main__':
    main()
