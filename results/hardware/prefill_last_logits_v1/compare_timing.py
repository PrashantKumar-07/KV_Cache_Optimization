"""Original ABBA harness; identical bulk policies, prefill logits only."""
import hashlib
from pathlib import Path
import sys
from candidate import compile_caller

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import compare_reuse_visible as driver
from compare_bulk_promotion import RecordedBulkPolicy
import latency_profile


class AllBulk(RecordedBulkPolicy):
    pass


class LastBulk(RecordedBulkPolicy):
    pass


def main():
    control = compile_caller('latency_profile.py', 'profile_one', latency_profile.__dict__, False)
    candidate = compile_caller('latency_profile.py', 'profile_one', latency_profile.__dict__, True)
    saved = driver.RecordedNativePolicy, driver.trial.FusedBoundedPolicy, driver.trial.measure,
    original_profile, original_write = driver.trial.profile_one, driver.atomic_json

    def measure(*args, **kwargs):
        driver.trial.profile_one = candidate if driver.trial.FusedBoundedPolicy is LastBulk else control
        try:
            return saved[2](*args, **kwargs)
        finally:
            driver.trial.profile_one = original_profile

    def write(path, report):
        report.update(actual_control='ordinary_bulk_all_prompt_logits',
            actual_candidate='ordinary_bulk_last_prompt_logits',
            logits_timing_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limits='Original recorded harness synchronization and policy-history instrumentation. '
                'Native prefill logit count is the sole factor; no physical STT/full resource equality claim.')
        original_write(path, report)

    driver.RecordedNativePolicy, driver.trial.FusedBoundedPolicy = LastBulk, AllBulk
    driver.trial.measure, driver.atomic_json = measure, write
    try:
        driver.main()
    finally:
        driver.RecordedNativePolicy, driver.trial.FusedBoundedPolicy, driver.trial.measure = saved
        driver.trial.profile_one, driver.atomic_json = original_profile, original_write


if __name__ == '__main__':
    main()
