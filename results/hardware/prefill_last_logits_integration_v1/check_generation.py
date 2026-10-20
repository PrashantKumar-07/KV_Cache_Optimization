"""Validate real opt-in generator API against archived ordinary control."""
from functools import partial
import importlib.util
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PRIOR = ROOT / 'results/hardware/prefill_last_logits_v1'
sys.path[:0] = [str(ROOT / 'experiments'), str(PRIOR)]
import longbench_eval as bench
spec = importlib.util.spec_from_file_location('prototype_generation', PRIOR / 'check_generation.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def main():
    original = driver.compile_caller

    def compile_actual(filename, function_name, namespace, candidate):
        if candidate:
            assert filename == 'longbench_eval.py' and function_name == 'generate_with_policy'
            return partial(bench.generate_with_policy, prefill_last_logits=True)
        return original(filename, function_name, namespace, False)

    driver.compile_caller = compile_actual
    try:
        driver.main()
    finally:
        driver.compile_caller = original


if __name__ == '__main__':
    main()
