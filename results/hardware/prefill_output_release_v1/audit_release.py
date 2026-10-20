"""Paired caller-only lifetime audit; archived control, unchanged bulk policy."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
spec = importlib.util.spec_from_file_location('frozen_owner_audit',
    ROOT / 'results/hardware/fresh_prefill_owner_v1/audit_fresh_owner.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
fresh = driver.audit.fresh


def main():
    index = sys.argv.index('--caller')
    caller = sys.argv[index + 1]
    assert caller in ('retained', 'released')
    del sys.argv[index:index + 2]
    original_main, original_write = fresh.main, fresh.atomic_json
    source = (HERE / 'measure_fresh_process_memory.py.before').read_text()
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    assert len(functions) == 1
    code = compile(ast.Module(body=functions, type_ignores=[]), str(HERE / 'archived_control'), 'exec')

    def retained():
        # Compile only when inner observers/policy/root hooks have been installed.
        namespace = fresh.__dict__.copy()
        exec(code, namespace)
        namespace['main']()

    def write(path, report):
        report.update(caller_output_lifetime=caller,
            release_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            archived_caller_sha256=hashlib.sha256(source.encode()).hexdigest())
        original_write(path, report)

    fresh.atomic_json = write
    if caller == 'retained':
        fresh.main = retained
    try:
        driver.main()
    finally:
        fresh.main, fresh.atomic_json = original_main, original_write


if __name__ == '__main__':
    main()
