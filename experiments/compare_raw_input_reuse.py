"""Frozen ordinary-bulk versus isolated raw-input payload-reuse harness."""
import hashlib
from pathlib import Path
import compare_bulk_promotion as comparison
from raw_input_reuse import RawInputPolicy, PROMOTE_AST, APPEND_AST, QUEUE_AST


class RecordedRawPolicy(comparison.RecordedBulkPolicy,RawInputPolicy):
    pass


def main():
    candidate = comparison.RecordedBulkPolicy
    baseline = comparison.driver.RecordedNativePolicy
    writer = comparison.driver.atomic_json
    def write(path,report):
        report.update(actual_control='ordinary_bulk_promotion',actual_candidate='raw_input_reuse',
            raw_input_reuse=True,raw_payload_ast=dict(promote=PROMOTE_AST,append=APPEND_AST,queue=QUEUE_AST),
            raw_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            raw_policy_sha256=hashlib.sha256(Path(__file__).with_name('raw_input_reuse.py').read_bytes()).hexdigest(),
            raw_limits='Research-only standardTorch per-layer scratch compaction. Read-only prefix validation charged in model path. No new native kernel/CLI/default. Mandatory peak/timing/lifetime/generation/stress gates remain.')
        writer(path,report)
    comparison.RecordedBulkPolicy = RecordedRawPolicy
    comparison.driver.RecordedNativePolicy = candidate
    comparison.driver.atomic_json = write
    try: comparison.main()
    finally:
        comparison.RecordedBulkPolicy = candidate
        comparison.driver.RecordedNativePolicy = baseline
        comparison.driver.atomic_json = writer


if __name__=='__main__': main()
