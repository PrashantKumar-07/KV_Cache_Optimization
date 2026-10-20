"""Research prefill-owner boundary and scoped interleaved-request gates."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments'))
from compare_bulk_promotion import RecordedBulkPolicy
from compare_prefill_owner import RecordedSharedPolicy
from forward_owner_release import OwnerRelease
from prefill_owner_share import PrefillSharedPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('boundary', 'interleaved'), required=True)
    parser.add_argument('--default-geometry', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = sorted(p for folder in ('src', 'experiments', 'tests')
                     for p in (ROOT / folder).iterdir() if p.suffix in ('.py', '.cu', '.cpp'))
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sources}
    details = []; original_argv = sys.argv
    if args.kind == 'boundary':
        import check_bounded_pinned_pool as driver
        driver.FusedBoundedPoolPolicy = PrefillSharedPolicy
        argv = ['--fused'] + (['--default-geometry'] if args.default_geometry else [])
    else:
        import check_cpu_interleaved_model as driver
        driver.ScorePolicy = RecordedBulkPolicy
        driver.FusedBoundedPolicy = RecordedSharedPolicy
        original_start, original_advance, original_finish = driver.start, driver.advance, driver.finish

        def start(model, *params, **options):
            state = original_start(model, *params, **options)
            state['lease_counts'] = dict(released_forwards=0, restored_forwards=0, released_layers=0)
            return state

        def advance(model, state):
            before = len(model._forward_pre_hooks), len(model._forward_hooks)
            lease = OwnerRelease(model, state['policy']) if isinstance(state['policy'], RecordedSharedPolicy) else None
            try:
                original_advance(model, state)
                if lease is not None:
                    assert lease.pending is None and lease.released_forwards == lease.restored_forwards
                    for name in state['lease_counts']:
                        state['lease_counts'][name] += getattr(lease, name)
            finally:
                if lease is not None:
                    lease.close()
                assert before == (len(model._forward_pre_hooks), len(model._forward_hooks))

        def finish(state):
            policy = state['policy']
            candidate = isinstance(policy, RecordedSharedPolicy)
            details.append(dict(candidate=candidate, steps=state['step'],
                prefill_shared_layers=getattr(policy, 'prefill_shared_layers', 0),
                raw_bound_layers=getattr(policy, 'raw_bound_layers', 0),
                raw_compactions=getattr(policy, 'raw_compactions', 0),
                lease_counts=state['lease_counts'].copy()))
            if candidate:
                steps = state['step']
                assert policy.prefill_shared_layers == 32
                assert policy.raw_bound_layers == policy.raw_compactions == steps * 32
                assert state['lease_counts'] == dict(released_forwards=steps,
                    restored_forwards=steps, released_layers=steps * 32)
            return original_finish(state)

        driver.start, driver.advance, driver.finish = start, advance, finish
        hot, cold = (1024, 2048) if args.default_geometry else (991, 1090)
        argv = ['--fused-bounded', '--hot', str(hot), '--cold', str(cold)]
    original_write = driver.atomic_json

    def write(path, report):
        report.update(actual_candidate='research_prefill_owner_share',
            preservation_kind=args.kind, request_ownership_checks=details,
            preservation_source_sha256=source_hashes,
            preservation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            preservation_limits='Boundary driver exercises multi-token fallbacks without model leases. '
                'Interleaved model execution attaches a lease only around the active request forward; '
                'no inactive-policy hooks. Serialized default stream; no asynchronous serving or timing claim.')
        original_write(path, report)

    driver.atomic_json = write
    sys.argv = ['check', '--output', str(args.output), *argv]
    try:
        driver.main()
    finally:
        driver.atomic_json = original_write
        sys.argv = original_argv


if __name__ == '__main__':
    main()
