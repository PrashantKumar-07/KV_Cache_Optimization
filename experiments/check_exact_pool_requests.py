"""Reuse native saturation/multi-pool/recycling gates with exact pinned roots."""
import argparse
import hashlib
from pathlib import Path
import sys
import check_reuse_requests as requests
from compare_reuse_visible import RecordedNativePolicy
from exact_pinned_pool import ExactSharedPool,ExactPinnedPoolPolicy
from exact_pinned_memory import stats


class RecordedExactPolicy(RecordedNativePolicy):
    direct_pool_cls=ExactSharedPool


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('native','boundary','interleaved'),required=True)
    parser.add_argument('--default-geometry',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    sources=list(Path('src').glob('*.py'))+list(Path('experiments').glob('*.py'))+list(Path('experiments').glob('*.cu'))+list(Path('experiments').glob('*.cpp'))
    identity=dict(exact_pinned_root=True,exact_pool_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    original_argv=sys.argv
    if args.kind=='native':
        import check_cpu_multi_pool as driver
        original=driver.DirectSharedPool;driver.DirectSharedPool=ExactSharedPool
        sys.argv=[str(Path(__file__)),'--bounded','--output',str(args.output)]
    else:
        import check_bounded_pinned_pool as driver
        if args.kind=='interleaved':import check_cpu_interleaved_model as driver
        original=requests.native.NativeReusePolicy
        recorded=requests.RecordedNativePolicy
        if args.kind=='boundary':
            requests.native.NativeReusePolicy=ExactPinnedPoolPolicy
        requests.RecordedNativePolicy=RecordedExactPolicy
        sys.argv=[str(Path(__file__)),'--kind',args.kind,'--output',str(args.output)]
        if args.default_geometry:sys.argv.append('--default-geometry')
    original_write=driver.atomic_json
    def write(path,report):
        report.update(identity,exact_allocation_stats=stats())
        original_write(path,report)
    driver.atomic_json=write
    try:
        if args.kind=='native':driver.main()
        else:requests.main()
        assert stats()['live_allocations']==0,'Exact root remains owned after reset'
    finally:
        sys.argv=original_argv;driver.atomic_json=original_write
        if args.kind=='native':driver.DirectSharedPool=original
        else:
            requests.native.NativeReusePolicy=original
            requests.RecordedNativePolicy=recorded


if __name__=='__main__':main()
