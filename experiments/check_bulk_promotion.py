"""Reuse frozen native, saturation, generation and recycling integrity drivers."""
import argparse
import hashlib
from pathlib import Path
import sys
import check_reuse_integrity as generation
import check_reuse_requests as requests
from bulk_promotion_policy import BulkPromotionPolicy
from compare_bulk_promotion import RecordedBulkPolicy
from native_reuse_visible import NativeReusePolicy


class NativeControl(generation.Trajectory,NativeReusePolicy):
    def __init__(self,**options):
        self.trajectory,self.frames=hashlib.sha256(),0
        super().__init__(**options)


class BulkGeneration(generation.Candidate,BulkPromotionPolicy):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('native','boundary','generation','interleaved'),required=True)
    parser.add_argument('--default-geometry',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.kind=='generation' and args.manifest is None:parser.error('--manifest required')
    files=list(Path('src').glob('*.py'))+list(Path('experiments').glob('*.py'))+list(Path('experiments').glob('*.cu'))+list(Path('experiments').glob('*.cpp'))
    identity=dict(bulk_promotion=True,actual_control='ordinary_native_reuse' if args.kind=='generation' else 'original_integrity_driver',
        actual_candidate='bulk_promotion_native_reuse',kind=args.kind,
        bulk_integrity_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    original_argv=sys.argv
    original_control,original_candidate=generation.Control,generation.Candidate
    original_recorded,original_native=requests.RecordedNativePolicy,requests.native.NativeReusePolicy
    if args.kind in ('native','generation'):
        driver=generation
        generation.Control,generation.Candidate=NativeControl,BulkGeneration
        argv=['--native-only'] if args.kind=='native' else ['--manifest',str(args.manifest)]
    else:
        if args.kind=='boundary':
            import check_bounded_pinned_pool as driver
            requests.native.NativeReusePolicy=BulkPromotionPolicy
        else:
            import check_cpu_interleaved_model as driver
            # Keep the original base so the existing gather-count hook reaches this subclass.
            requests.RecordedNativePolicy=RecordedBulkPolicy
        argv=['--kind',args.kind]
        if args.default_geometry:argv.append('--default-geometry')
    original_write=driver.atomic_json
    def write(path,report):
        report.update(identity)
        original_write(path,report)
    driver.atomic_json=write
    sys.argv=['check','--output',str(args.output),*argv]
    try:
        (generation.main if args.kind in ('native','generation') else requests.main)()
    finally:
        sys.argv=original_argv;driver.atomic_json=original_write
        generation.Control,generation.Candidate=original_control,original_candidate
        requests.RecordedNativePolicy,requests.native.NativeReusePolicy=original_recorded,original_native


if __name__=='__main__':main()
