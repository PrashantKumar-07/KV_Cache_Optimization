"""Untimed line allocation audit; unchanged ordinary bulk/last-logit caller."""
import hashlib
import importlib.util
import json
import linecache
from pathlib import Path
import sys
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import longbench_eval as bench
from tiered_kv_cache import TieredKVCache
from direct_shared_pool import DirectCpuCache
from cpu_batched_cache import SharedPool
spec = importlib.util.spec_from_file_location('integrated_audit',
    ROOT / 'results/hardware/prefill_last_logits_integration_v1/audit_resources.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
fresh = driver.driver.audit.fresh


def main():
    saved_trace, saved_write = sys.gettrace(), fresh.atomic_json
    events = []; active = {}; sequence = {}; bifurcating = False
    caller = fresh.main.__wrapped__.__code__
    codes = {caller: 'caller', bench.TieredKVPolicy._bifurcate.__code__: 'bifurcate',
        bench.gqa_causal_importance.__code__: 'importance',
        TieredKVCache.initial_bifurcation.__code__: 'placement',
        DirectCpuCache.initial_bifurcation.__code__: 'direct_init',
        SharedPool.enqueue.__code__: 'enqueue', SharedPool.flush.__code__: 'flush'}
    caller_limit = next(number for number, line in enumerate(Path(caller.co_filename).read_text().splitlines(), 1)
        if line.strip() == 'predictions=[]')

    def trace(frame, event, arg):
        nonlocal bifurcating
        code = frame.f_code
        if code not in codes:
            return None
        name = codes[code]
        if event == 'call':
            sequence[name] = sequence.get(name, 0) + 1
            active[id(frame)] = sequence[name]
            if name == 'bifurcate':
                bifurcating = True
        observed = (name == 'caller' and frame.f_lineno <= caller_limit and
            'ids' in frame.f_locals and frame.f_locals['ids'].device.type == 'cuda') or (
            name != 'caller' and bifurcating)
        if observed and event in ('line', 'return'):
            torch.cuda.synchronize()
            locals_ = frame.f_locals
            tensors = {}
            for key in ('k_2d', 'v_2d', 'q_real', 'q_window', 'k_all', 'v_all', 'scores', 'received', 'importance', 'packed'):
                tensor = locals_.get(key)
                if isinstance(tensor, torch.Tensor):
                    tensors[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                        storage_bytes=tensor.untyped_storage().nbytes())
            events.append(dict(function=name, call=active[id(frame)], event=event,
                line=frame.f_lineno, source=linecache.getline(code.co_filename, frame.f_lineno).strip(),
                layer=locals_.get('layer_idx'), allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(), tensors=tensors))
        if event == 'return':
            active.pop(id(frame), None)
            if name == 'bifurcate':
                bifurcating = False
        return trace

    def write(path, report):
        report.update(allocation_trace=events, allocation_trace_counts=sequence.copy(),
            allocation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            allocation_limits='sys.settrace and CUDA synchronization invalidate timings and can change deferred lifetimes. No source, dtype, selection, caller peak-reset or payload ownership change. Tensor descriptors and primitive counters only.')
        saved_write(path, report)

    fresh.atomic_json = write
    sys.settrace(trace)
    try:
        driver.main()
    finally:
        sys.settrace(saved_trace)
        fresh.atomic_json = saved_write
        (BASE / 'cleanup.json').write_text(json.dumps(dict(trace_restored=sys.gettrace() is saved_trace,
            writer_restored=fresh.atomic_json is saved_write, active_frames=len(active)), indent=2) + '\n')


if __name__ == '__main__':
    main()
