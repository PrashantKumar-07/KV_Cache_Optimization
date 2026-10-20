"""Read-only primitive cache-width observation in the frozen ordinary fresh caller."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
prior=ROOT/'results/hardware/layerwise_prefill_placement_v1'
sys.path.insert(0,str(prior))
spec=importlib.util.spec_from_file_location('frozen_ordinary_resource_driver',prior/'audit_resources.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def main():
    original=(driver.audit.fresh.AutoModelForCausalLM,driver.audit.fresh.atomic_json,sys.argv)
    output=BASE/'cpu_gpu.json';assert not output.exists()
    models=[];handles=[];baselines=[];observations=[]
    def before(model,args,options):
        ids=options.get('input_ids',args[0] if args else None)
        past=options.get('past_key_values')
        observations.append(dict(input_rows=ids.shape[1],
            before_cache_rows=[layer.keys.shape[-2] for layer in past.layers] if past is not None else [],
            before_cache_heads=[layer.keys.shape[1] for layer in past.layers] if past is not None else []))
    class Loader:
        @staticmethod
        def from_pretrained(*args,**options):
            model=original[0].from_pretrained(*args,**options)
            models.append(model);baselines.append((len(model._forward_pre_hooks),len(model._forward_hooks)))
            handles.append(model.register_forward_pre_hook(before,with_kwargs=True))
            return model
    def write(path,report):
        report.update(ordinary_width_observations=observations,
            ordinary_width_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            width_observer_limits='Primitive root pre-hook; no stored GPU tensors/math/reset changes. Cleanup verified separately after nested driver restores its baseline. Times invalid.')
        original[1](path,report)
    driver.audit.fresh.AutoModelForCausalLM=Loader;driver.audit.fresh.atomic_json=write
    sys.argv=['audit','--arm','control','--output',str(output)]
    try:driver.main()
    finally:
        for handle in handles:handle.remove()
        if models:assert baselines[0]==(len(models[0]._forward_pre_hooks),len(models[0]._forward_hooks))
        driver.audit.fresh.AutoModelForCausalLM,driver.audit.fresh.atomic_json,sys.argv=original
        path=BASE/'cpu_width_cleanup.json';assert not path.exists()
        path.write_text(json.dumps(dict(root_width_hook_removed=True,nested_driver_restored=True),indent=2)+'\n')


if __name__=='__main__':main()
