"""Fresh-process reuse/control memory using the unchanged measurement driver."""
import argparse
import hashlib
from pathlib import Path
import sys
import torch
import measure_fresh_process_memory as driver
from audit_pinned_blocks import kernel
from fused_bounded_pool import FusedBoundedPoolPolicy
from native_reuse_visible import NativeReusePolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('fused_bounded','native_reuse'),required=True)
    parser.add_argument('--hot',type=int,required=True)
    parser.add_argument('--cold',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    module=kernel()
    original_roots,original_policy=driver.PinnedRoots,driver.BoundedPoolPolicy
    original_memory,original_write,original_argv=driver.memory,driver.atomic_json,sys.argv
    phase=dict(whole_allocated_peak=0,whole_reserved_peak=0,prefill_peak=0,decode_peak=0)
    root_tracker={}
    sources=list(Path('experiments').glob('*.py'))+list(Path('experiments').glob('*.cpp'))+list(Path('experiments').glob('*.cu'))+list(Path('src').glob('*.py'))
    identity=dict(actual_arm=args.arm,host_block_probe_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),host_block_probe_method=module.method,
        actual_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    class BlockRoots(original_roots):
        def __init__(self):
            super().__init__()
            self.blocks=[];self.block_sizes={};self.block_current=self.block_peak=0
            root_tracker['tracker']=self

        def track(self,tensor):
            super().track(tensor)
            info=module.block_info(tensor)
            size=info['requested_storage_bytes'];capacity=info['allocator_block_bytes']
            self.block_sizes[size]=capacity
            self.block_current+=capacity;self.block_peak=max(self.block_peak,self.block_current)
            self.blocks.append(dict(**info,owned_block_capacity_bytes=self.block_current))

        def release(self,size):
            super().release(size)
            self.block_current-=self.block_sizes[size]

    def memory(stage):
        result=original_memory(stage)
        phase['whole_allocated_peak']=max(phase['whole_allocated_peak'],result['gpu_peak_allocated_bytes'])
        phase['whole_reserved_peak']=max(phase['whole_reserved_peak'],result['gpu_peak_reserved_bytes'])
        if stage=='prefill_compressed':
            phase['prefill_peak']=result['gpu_peak_allocated_bytes']
            torch.cuda.reset_peak_memory_stats()
        elif stage.startswith('decode_') or stage=='generation_complete':
            phase['decode_peak']=max(phase['decode_peak'],result['gpu_peak_allocated_bytes'])
        result['phase_gpu_peak_allocated_bytes']=result['gpu_peak_allocated_bytes']
        result['gpu_peak_allocated_bytes']=phase['whole_allocated_peak']
        result['gpu_peak_reserved_bytes']=phase['whole_reserved_peak']
        tracker=root_tracker.get('tracker')
        result['owned_host_allocator_block_bytes']=tracker.block_current if tracker else 0
        return result

    def write(path,report):
        tracker=root_tracker.get('tracker')
        report.update(identity,gpu_phase_peaks=phase.copy())
        if tracker:
            report.update(host_block_observations=tracker.blocks,
                owned_host_allocator_block_peak_bytes=tracker.block_peak,
                owned_host_allocator_block_after_cleanup_bytes=tracker.block_current)
        report['actual_limitations']='Live owned pinned allocation ranges measured through the public CUDA driver API. Excludes other cached host blocks and GPU-driver allocations; RSS samples lowerbound and VmHWM includes imports/model load. GPU allocated/reserved peaks measured separately for prefill/decode. Same old fresh-process fixed teacher stream, not timing/quality/physical STT or resource equality proof.'
        original_write(path,report)
    driver.PinnedRoots=BlockRoots
    driver.BoundedPoolPolicy=FusedBoundedPoolPolicy if args.arm=='fused_bounded' else NativeReusePolicy
    driver.memory=memory;driver.atomic_json=write
    sys.argv=[str(Path(__file__)),'--arm','cpu_bounded','--hot',str(args.hot),'--cold',str(args.cold),'--output',str(args.output)]
    try:driver.main()
    finally:
        driver.PinnedRoots,driver.BoundedPoolPolicy=original_roots,original_policy
        driver.memory,driver.atomic_json,sys.argv=original_memory,original_write,original_argv


if __name__=='__main__':main()
