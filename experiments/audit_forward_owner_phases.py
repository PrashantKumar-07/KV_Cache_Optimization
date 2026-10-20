"""Public allocator counters before/after forward ownership release."""
import hashlib
from pathlib import Path
import torch
import compare_forward_owner as comparison
import audit_forward_owner as audit


def memory(stage,**extra):
    return dict(stage=stage,allocated_bytes=torch.cuda.memory_allocated(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),**extra)


class ObservedOwner(comparison.OwnerRelease):
    frames=[]

    def record(self,stage,args):
        type(self).frames.append(memory(stage,input_rows=args[0].shape[1],
            released_forwards=self.released_forwards,restored_forwards=self.restored_forwards,
            pending=self.pending is not None))

    def before(self,model,args,kwargs):
        self.record('lease_before',args)
        result=super().before(model,args,kwargs)
        self.record('lease_after',args)
        return result

    def after(self,model,args,kwargs,out):
        self.record('forward_before_restore',args)
        result=super().after(model,args,kwargs,out)
        self.record('forward_after_restore',args)
        return result


def main():
    driver=comparison.comparison.comparison.driver
    previous=comparison.OwnerRelease,driver.atomic_json
    ObservedOwner.frames=[]
    def write(path,report):
        report.update(owner_phase_diagnostic=True,owner_phase_frames=ObservedOwner.frames,
            owner_phase_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            owner_phase_limits='Read-only public allocator counters; no peak reset or GPU payload owners. One4K context; timing invalid. Failed memory primary unchanged.')
        previous[1](path,report)
    comparison.OwnerRelease=ObservedOwner;driver.atomic_json=write
    try:audit.main()
    finally:comparison.OwnerRelease,driver.atomic_json=previous


if __name__=='__main__':main()
