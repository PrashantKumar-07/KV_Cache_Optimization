"""Existing native-reuse model harness; ordinary versus bulk CPU planning."""
import hashlib
from pathlib import Path
import compare_reuse_visible as driver
from bulk_promotion_policy import BulkPromotionPolicy, PREPARE_AST


class RecordedBulkPolicy(driver.RecordedNativePolicy,BulkPromotionPolicy):
    pass


def main():
    original = driver.RecordedNativePolicy
    old_control = driver.trial.FusedBoundedPolicy
    old_write = driver.atomic_json

    def write(path,report):
        report.update(actual_control='ordinary_native_reuse',actual_candidate='bulk_promotion_native_reuse',
            prepare_ast=PREPARE_AST,comparison_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            promotion_candidate_mro=[c.__name__ for c in RecordedBulkPolicy.__mro__],
            stage_scope='Original recorded harness; legacy arm names identify actual control/candidate above. Default host allocator, no prefill cleanup/expandable layout. No ordinary CLI change.')
        old_write(path,report)

    driver.RecordedNativePolicy = RecordedBulkPolicy
    driver.trial.FusedBoundedPolicy = original
    driver.atomic_json = write
    try: driver.main()
    finally:
        driver.RecordedNativePolicy = original
        driver.trial.FusedBoundedPolicy = old_control
        driver.atomic_json = old_write


if __name__ == '__main__': main()
