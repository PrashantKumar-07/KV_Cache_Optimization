"""Primitive prefill model-layer allocation audit of the ordinary caller."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from transformers.utils.output_capturing import maybe_install_capturing_hooks

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
spec = importlib.util.spec_from_file_location('integrated_resource_audit',
    ROOT / 'results/hardware/prefill_last_logits_integration_v1/audit_resources.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
fresh = driver.driver.audit.fresh


def main():
    original_loader, original_write = fresh.AutoModelForCausalLM, fresh.atomic_json
    handles = []; events = []; hooked = []

    def record(label, phase, module, positional, options, out=None):
        hidden = options.get('hidden_states', positional[0] if positional else None)
        if hidden is None or hidden.ndim != 3 or hidden.shape[1] <= 1:
            return
        torch.cuda.synchronize()
        roots = {}
        cache = options.get('past_key_values')
        if cache is not None:
            for layer in cache.layers:
                for tensor in (layer.keys, layer.values):
                    if tensor is not None:
                        storage = tensor.untyped_storage()
                        roots[storage.data_ptr()] = storage.nbytes()
        events.append(dict(label=label, phase=phase, input_shape=list(hidden.shape),
            input_dtype=str(hidden.dtype), output_shape=list(out.shape) if isinstance(out, torch.Tensor) else None,
            cache_unique_bytes=sum(roots.values()) if cache is not None else None,
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))

    class Loader:
        @staticmethod
        def from_pretrained(*args, **options):
            model = original_loader.from_pretrained(*args, **options)
            # Native output capture installs persistent child hooks on first decode.
            maybe_install_capturing_hooks(model.model)
            for index, block in enumerate(model.model.layers):
                modules = dict(block=block, attention=block.self_attn, mlp=block.mlp,
                    gate=block.mlp.gate_proj, up=block.mlp.up_proj, down=block.mlp.down_proj)
                for name, module in modules.items():
                    label = f'{index}.{name}'
                    hooked.append((label, module, len(module._forward_pre_hooks), len(module._forward_hooks)))
                    def before(module, positional, options, label=label):
                        record(label, 'before', module, positional, options)
                    def after(module, positional, options, out, label=label):
                        record(label, 'after', module, positional, options, out)
                    handles.extend((module.register_forward_pre_hook(before, with_kwargs=True),
                        module.register_forward_hook(after, with_kwargs=True)))
            return model

    def write(path, report):
        report.update(model_allocation_events=events,
            model_allocation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_allocation_limits='Primitive child-hook observations; no output/cache/frame storage retained. CUDA synchronization invalidates timings; one model/seed/16K fixed stream, no optimization or resource-equality claim.')
        original_write(path, report)

    fresh.AutoModelForCausalLM, fresh.atomic_json = Loader, write
    try:
        driver.main()
    finally:
        for handle in handles:
            handle.remove()
        restored = all((len(module._forward_pre_hooks), len(module._forward_hooks)) == (pre, post)
            for label, module, pre, post in hooked)
        fresh.AutoModelForCausalLM, fresh.atomic_json = original_loader, original_write
        (BASE / 'model_v3_cleanup.json').write_text(json.dumps(dict(child_hooks_restored=restored,
            hook_count=len(handles), loader_restored=fresh.AutoModelForCausalLM is original_loader,
            writer_restored=fresh.atomic_json is original_write, mismatches=[dict(label=label, before=[pre,post], after=[len(module._forward_pre_hooks),len(module._forward_hooks)]) for label,module,pre,post in hooked if (len(module._forward_pre_hooks),len(module._forward_hooks)) != (pre,post)]), indent=2) + '\n')
        assert restored


if __name__ == '__main__':
    main()
