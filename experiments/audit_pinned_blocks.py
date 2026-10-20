"""Read live pinned allocation ranges through the public CUDA driver API."""
import argparse
import ctypes
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from provenance import atomic_json, runtime_identity


@lru_cache(maxsize=1)
def kernel():
    driver=ctypes.CDLL('libcuda.so.1')
    query=driver.cuPointerGetAttribute
    query.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64]
    query.restype=ctypes.c_int
    def block_info(tensor):
        if tensor.device.type!='cpu' or not tensor.is_pinned():
            raise ValueError('Expected a live pinned CPU storage')
        storage=tensor.untyped_storage();pointer=storage.data_ptr()
        values={}
        for attribute,name in ((2,'memory_type'),(11,'range_start'),(12,'range_bytes')):
            value=ctypes.c_uint32() if attribute==2 else ctypes.c_uint64()
            code=query(ctypes.byref(value),attribute,pointer)
            if code:raise RuntimeError(f'CUDA pointer attribute{attribute} failed with status{code}')
            values[name]=value.value
        if values['memory_type']!=1 or values['range_start']!=pointer or values['range_bytes']<storage.nbytes():
            raise ValueError('Unexpected host allocation range')
        return dict(requested_storage_bytes=storage.nbytes(),allocator_block_bytes=values['range_bytes'],
            allocated=True,event_count=None,public_cuda_memory_type=values['memory_type'])
    return SimpleNamespace(block_info=block_info,__file__=__file__,method='public_cuPointerGetAttribute')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare-private',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=dict(passed=False,runtime=runtime_identity(),checks=[],probe_method='public_cuPointerGetAttribute',
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        module=kernel();report['probe_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        private=None
        if args.compare_private:
            binary=Path('/data/nishant/tieredkv_external/host_block_audit_build_20261002/tieredkv_host_block_audit_v1.so')
            proof=json.loads(Path('results/hardware/actual_reuse_memory_20261002/block_preflight.json').read_text())
            assert proof['passed'] and hashlib.sha256(binary.read_bytes()).hexdigest()==proof['binary_sha256']
            spec=importlib.util.spec_from_file_location('tieredkv_host_block_audit_v1',binary)
            private=importlib.util.module_from_spec(spec);spec.loader.exec_module(private)
            report['private_crosscheck_binary_sha256']=proof['binary_sha256']
        for size in (1,31,32,33,151257088,276824064):
            tensor=torch.empty(size,dtype=torch.uint8,pin_memory=True)
            result=module.block_info(tensor)
            assert result['requested_storage_bytes']==size
            assert result['allocator_block_bytes']==1<<(size-1).bit_length()
            view=tensor[1:] if size>1 else tensor[:]
            assert module.block_info(view)==result
            cross=private.block_info(tensor)['allocator_block_bytes']==result['allocator_block_bytes'] if private else None
            if private:assert cross
            report['checks'].append(dict(bytes=size,view_storage_alias_exact=True,private_capacity_exact=cross,**result))
            del tensor,view;torch._C._host_emptyCache()
        try:module.block_info(torch.empty(16))
        except ValueError:report['nonpinned_rejection']=True
        else:raise AssertionError('Unpinned storage accepted')
        report.update(passed=True,limitations='Public CUDA allocation-address-range query, validated on live pinned Torch roots/aliases and optionally against the archived private reader. Reports allocator-requested host range, not OS page rounding, process-wide cached host totals, GPU workspaces/RSS/physical STT. No custom extension, compiler or private allocator layout required.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)


if __name__=='__main__':main()
