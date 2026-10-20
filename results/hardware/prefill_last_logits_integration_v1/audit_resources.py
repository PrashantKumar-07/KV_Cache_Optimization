"""Validate the real fresh-caller CLI option against frozen prototypes."""
import importlib.util
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
spec = importlib.util.spec_from_file_location('ordinary_bulk_audit',
    ROOT / 'results/hardware/fresh_prefill_owner_v1/audit_fresh_owner.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def main():
    original = driver.audit.fresh.main

    def run():
        saved = sys.argv
        sys.argv = [*saved, '--prefill-last-logits']
        try:
            original()
        finally:
            sys.argv = saved

    driver.audit.fresh.main = run
    try:
        driver.main()
    finally:
        driver.audit.fresh.main = original


if __name__ == '__main__':
    main()
