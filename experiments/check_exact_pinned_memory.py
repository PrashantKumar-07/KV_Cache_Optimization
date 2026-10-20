"""Exact host ranges, alias owners and delayed same-context transfer lifetime."""
import argparse
import gc
import hashlib
from pathlib import Path
import torch
from audit_pinned_blocks import kernel as range_probe
from exact_pinned_memory import allocate,stats
from provenance import atomic_json,runtime_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=dict(passed=False,runtime=runtime_identity(),checks=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        map(Path,('experiments/exact_pinned_memory.py','experiments/check_exact_pinned_memory.py','experiments/audit_pinned_blocks.py'))})
    try:
        torch.cuda.set_device(0)
        for size in (1,31,32,33,151257088,276824064):
            tensor=allocate((size,),dtype=torch.uint8)
            info=range_probe().block_info(tensor)
            assert info['requested_storage_bytes']==info['allocator_block_bytes']==size
            tensor[:min(size,32)].fill_(7)
            view=tensor[:min(size,32)];del tensor;gc.collect()
            assert stats()['live_requested_bytes']==size and torch.all(view==7)
            del view;gc.collect()
            assert stats()['live_allocations']==0
            report['checks'].append(dict(case='exact_range_alias_owner',bytes=size,exact=True))
        expected=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1,8,64,128)
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for repeat in range(12):
            stream=streams[repeat%2]
            host=allocate(expected.shape);host.copy_(expected)
            with torch.cuda.stream(stream):
                torch.cuda._sleep(40000000)
                copied=host.to('cuda',non_blocking=True)
            del host;gc.collect()  # Finalizer must wait for the queued copy.
            assert stats()['live_allocations']==0
            assert torch.equal(copied.cpu().contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
            report['checks'].append(dict(case='delayed_h2d_owner_release',repeat=repeat,consumer=repeat%2,exact=True))
        host=allocate(expected.shape)
        with torch.cuda.stream(streams[0]):
            torch.cuda._sleep(40000000)
            host.copy_(expected.cuda(),non_blocking=True)
        view=host[:];del host;gc.collect()
        streams[0].synchronize()
        assert torch.equal(view.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
        del view;gc.collect();assert stats()['live_allocations']==0
        report['checks'].append(dict(case='delayed_d2h_last_view_owner',exact=True))
        report.update(passed=True,allocation_events=stats()['events'],limitations='Direct CUDA host ranges and BF16 byte preservation. Same owner CUDA context/two streams; explicit context-wide synchronization at final owner release. No mixed-context/multi-GPU consumers, serving throughput, complete process/OS memory or physical STT claim. Native shared-pool transfer and model gates remain mandatory.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)


if __name__=='__main__':main()
