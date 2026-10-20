"""Same layout driver with the established timing observer in both rows."""
import audit_allocator_layout as layout
import compare_cpu_batched as trial


def main():
    original = trial.measure

    def measure(*args, **options):
        assert len(args) == 9 and args[8] is True, 'Expected the frozen two-row pilot call'
        # Same history/synchronization observation as earlier ABBA timing trials.
        result = original(*args[:8], False, **options)
        result['complete_state_hash_disabled_for_timing'] = True
        return result

    trial.measure = measure
    try:
        layout.main()
    finally:
        trial.measure = original


if __name__ == '__main__':
    main()
