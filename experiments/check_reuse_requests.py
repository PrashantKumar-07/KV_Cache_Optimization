"""Run unchanged saturation/recycling drivers with the native reuse policy."""
import argparse
import hashlib
from pathlib import Path
import sys
import native_reuse_visible as native
from compare_reuse_visible import RecordedNativePolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('boundary','interleaved'), required=True)
    parser.add_argument('--default-geometry', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kind=='boundary':
        import check_bounded_pinned_pool as driver
        original_policy = driver.FusedBoundedPoolPolicy
        driver.FusedBoundedPoolPolicy = native.NativeReusePolicy
        argv = ['--fused'] + (['--default-geometry'] if args.default_geometry else [])
    else:
        import check_cpu_interleaved_model as driver
        original_policy = driver.FusedBoundedPolicy
        driver.FusedBoundedPolicy = RecordedNativePolicy
        hot,cold = (1024,2048) if args.default_geometry else (991,1090)
        argv = ['--fused-bounded','--hot',str(hot),'--cold',str(cold)]
    original_atomic, original_gather, original_argv = driver.atomic_json, native.NativeReusePolicy.hot_gather, sys.argv
    counts = dict(calls=0,layers=0)
    sources = list(Path('src').glob('*.py')) + list(Path('experiments').glob('*.py')) + list(Path('experiments').glob('*.cu')) + list(Path('experiments').glob('*.cpp'))
    identity = dict(native_reuse=True, reuse_binary_sha256=hashlib.sha256(Path(native.kernel().__file__).read_bytes()).hexdigest(),
        reuse_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    def observed_gather(states,destinations):
        result = original_gather(states,destinations)
        counts['calls'] += 1; counts['layers'] += len(states)
        return result
    def write(path,report):
        report.update(identity, native_reuse_launches=counts.copy())
        original_atomic(path,report)
    native.NativeReusePolicy.hot_gather = staticmethod(observed_gather)
    driver.atomic_json = write
    sys.argv = [str(Path(__file__)), '--output', str(args.output), *argv]
    try:
        driver.main()
        assert counts['calls']>0 and counts['layers']>0
    finally:
        sys.argv = original_argv; driver.atomic_json = original_atomic
        native.NativeReusePolicy.hot_gather = staticmethod(original_gather)
        if args.kind=='boundary':driver.FusedBoundedPoolPolicy = original_policy
        else:driver.FusedBoundedPolicy = original_policy


if __name__=='__main__':main()
