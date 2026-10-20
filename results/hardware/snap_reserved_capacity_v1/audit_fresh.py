"""Frozen fresh Snap caller with genuine reserved adapter storage only."""
import functools
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from candidate import ReservedSnapKV
spec=importlib.util.spec_from_file_location('frozen_snap_fresh_audit',ROOT/'results/hardware/snap_actual_envelope_v1/audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    original=audit.SnapKVAdapter,audit.atomic_json,sys.argv
    weak={}
    class ObservedReserved(ReservedSnapKV):
        def update(self,*args,**kwargs):
            result=super().update(*args,**kwargs)
            for pair in self.roots.values():
                for tensor in pair:
                    storage=tensor.untyped_storage();key=storage._cdata
                    if key not in weak:weak[key]=storage._weak_ref()
            return result
    def write(path,report):
        gc.collect();torch.cuda.synchronize()
        expired=sum(torch.UntypedStorage._expired(ref) for ref in weak.values())
        if report['passed']:assert expired==len(weak)==64,'Reserved KV roots still owned'
        report.update(reserved_capacity_rows=1024,reserved_root_count=len(weak),reserved_roots_expired=expired,
            reserved_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reserved_limits='Same frozen fresh audit, original upstream prompt selector with fixed genuine future KV slots. '
                'Populated-prefix attention and expanded heads preserved. Research baseline only; no speed, '
                'exact TieredKV-use/width match or full512MiB/quality/physicalSTT claim.')
        original[1](path,report)
    audit.SnapKVAdapter=functools.partial(ObservedReserved,max_rows=1024);audit.atomic_json=write
    sys.argv=['audit','--output',str(BASE/'fresh.json')]
    try:audit.main()
    finally:
        for ref in weak.values():torch.UntypedStorage._free_weak_ref(ref)
        audit.SnapKVAdapter,audit.atomic_json,sys.argv=original
        path=BASE/'fresh_cleanup.json';assert not path.exists()
        path.write_text(json.dumps(dict(factory_writer_argv_restored=True,weak_handles_released=True),indent=2)+'\n')


if __name__=='__main__':main()
