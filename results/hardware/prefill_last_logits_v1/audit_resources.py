"""Fresh paired resources for last-prompt-logit allocation only."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from candidate import compile_caller

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
spec = importlib.util.spec_from_file_location('ordinary_audit',
    ROOT / 'results/hardware/fresh_prefill_owner_v1/audit_fresh_owner.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
fresh = driver.audit.fresh


def main():
    index = sys.argv.index('--logits')
    arm = sys.argv[index + 1]
    assert arm in ('all', 'last')
    del sys.argv[index:index + 2]
    original_main, original_write = fresh.main, fresh.atomic_json

    def run():
        compile_caller('measure_fresh_process_memory.py', 'main', fresh.__dict__, arm == 'last')()

    def write(path, report):
        report.update(prefill_logits_arm=arm,
            logits_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            logits_transform_sha256=hashlib.sha256((BASE / 'candidate.py').read_bytes()).hexdigest())
        original_write(path, report)

    fresh.main, fresh.atomic_json = run, write
    try:
        driver.main()
    finally:
        fresh.main, fresh.atomic_json = original_main, original_write


if __name__ == '__main__':
    main()
