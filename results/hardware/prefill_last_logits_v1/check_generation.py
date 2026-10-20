"""All-prompt versus last-prompt logits with identical ordinary bulk policy."""
import hashlib
from pathlib import Path
import sys
from candidate import compile_caller

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import check_reuse_integrity as driver
import check_batched_generation as generation
from bulk_promotion_policy import BulkPromotionPolicy


class Control(driver.Trajectory, BulkPromotionPolicy):
    def __init__(self, **options):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)


class Candidate(Control):
    counts = {}


def main():
    namespace = generation.bench.__dict__
    control = compile_caller('longbench_eval.py', 'generate_with_policy', namespace, False)
    candidate = compile_caller('longbench_eval.py', 'generate_with_policy', namespace, True)
    saved = driver.Control, driver.Candidate, driver.measure, generation.bench.generate_with_policy

    def measure(*args, **options):
        generation.bench.generate_with_policy = control if options['policy_cls'] is Control else candidate
        try:
            return saved[2](*args, **options)
        finally:
            generation.bench.generate_with_policy = saved[3]

    driver.Control, driver.Candidate, driver.measure = Control, Candidate, measure
    try:
        driver.main()
    finally:
        driver.Control, driver.Candidate, driver.measure, generation.bench.generate_with_policy = saved


if __name__ == '__main__':
    main()
