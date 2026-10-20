"""Weak-storage observation of prefill logits through the first decode forward."""
import hashlib
from pathlib import Path
import torch
import measure_fresh_process_memory as driver
import measure_actual_reuse_memory as actual


def main():
    original_loader,original_write=driver.AutoModelForCausalLM,driver.atomic_json
    rows=[];handles=[];prefill={}
    def alive():
        return bool(prefill) and not torch.UntypedStorage._expired(prefill['weak'])
    def before(model,args,kwargs):
        rows.append(dict(call=len(rows)+1,prefill_logits_alive_before_forward=alive()))
    def after(model,args,kwargs,result):
        logits=result.logits
        storage=logits.untyped_storage()
        if len(rows)==1:prefill['weak']=storage._weak_ref()
        rows[-1].update(logits_shape=list(logits.shape),logits_dtype=str(logits.dtype),
            logits_storage_bytes=storage.nbytes(),prefill_logits_alive_after_forward=alive())
    class Loader:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            model=original_loader.from_pretrained(*args,**kwargs)
            handles.append(model.register_forward_pre_hook(before,with_kwargs=True))
            handles.append(model.register_forward_hook(after,with_kwargs=True))
            return model
    def write(path,report):
        report.update(logits_lifetime_rows=rows,
            logits_audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            logits_audit_limitations='Weak C++ StorageImpl handles retain no tensor/payload owners; callback records primitive shapes/bytes/aliveness only. No local-frame inspection or altered model arguments. Untimed diagnostic, not an effect estimate.')
        original_write(path,report)
    driver.AutoModelForCausalLM=Loader;driver.atomic_json=write
    try:actual.main()
    finally:
        for handle in handles:handle.remove()
        if prefill:torch.UntypedStorage._free_weak_ref(prefill['weak'])
        driver.AutoModelForCausalLM,driver.atomic_json=original_loader,original_write


if __name__=='__main__':main()
