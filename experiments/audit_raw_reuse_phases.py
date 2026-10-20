"""Untimed execution/allocator phase observations after a failed memory pilot."""
import hashlib
from pathlib import Path
import torch
import compare_raw_input_reuse as comparison
import audit_raw_input_reuse as resources


def memory(stage,**extra):
    return dict(stage=stage,allocated_bytes=torch.cuda.memory_allocated(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),**extra)


class ObservedRaw(comparison.RecordedRawPolicy):
    frames=[]

    def record(self,stage):
        type(self).frames.append(memory(stage,step=self.decode_step_count,
            bound_layers=self.raw_bound_layers,compactions=self.raw_compactions))

    def _prepare_decode_reads(self):
        self.record('prepare_entry')
        result=super()._prepare_decode_reads()
        self.record('prepare_exit')
        return result

    def _visible_tuple(self,past):
        result=super()._visible_tuple(past)
        self.record('visible_exit')
        return result


def main():
    driver=comparison.comparison.driver
    original=(comparison.RecordedRawPolicy,driver.AutoModelForCausalLM,driver.atomic_json)
    ObservedRaw.frames=[];forwards=[];handles=[]
    class Loader:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            model=original[1].from_pretrained(*args,**kwargs)
            def after(module,args,kwargs,out):
                forwards.append(memory('forward_exit',call=len(forwards)+1,
                    input_rows=args[0].shape[1]))
            handles.append(model.register_forward_hook(after,with_kwargs=True))
            return model
    def write(path,report):
        report.update(raw_phase_diagnostic=True,raw_phase_frames=ObservedRaw.frames,
            raw_forward_frames=forwards,raw_phase_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            raw_phase_limits='Read-only public allocator counters; no peak reset or retained GPU payload owner. Observer timing invalid. Counters confirm branch execution;4K-only phase attribution. No advancement after failed primary.')
        original[2](path,report)
    comparison.RecordedRawPolicy=ObservedRaw
    driver.AutoModelForCausalLM,driver.atomic_json=Loader,write
    try:resources.main()
    finally:
        for handle in handles:handle.remove()
        comparison.RecordedRawPolicy,driver.AutoModelForCausalLM,driver.atomic_json=original


if __name__=='__main__':main()
