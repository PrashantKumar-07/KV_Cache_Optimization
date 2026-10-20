"""Archived/current generator parity with identical ordinary bulk policy."""
import ast
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
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
    source = (HERE / 'longbench_eval.py.before').read_text()
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
        and n.name == 'generate_with_policy']
    assert len(functions) == 1
    namespace = generation.bench.__dict__.copy()
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'archived_generator', 'exec'), namespace)
    saved = driver.Control, driver.Candidate, driver.measure, generation.bench.generate_with_policy

    def measure(*args, **options):
        generation.bench.generate_with_policy = namespace['generate_with_policy'] if options['policy_cls'] is Control else saved[3]
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
