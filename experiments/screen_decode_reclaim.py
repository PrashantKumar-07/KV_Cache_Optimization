"""Isolated charged decode-cache-release screen; research caller only."""
import ast
import hashlib
import inspect
import os
from pathlib import Path
import torch
import latency_profile
import compare_cpu_batched as trial
import screen_prefill_reclaim as screen


def profiles():
    tree = ast.parse(inspect.getsource(latency_profile.profile_one))
    original = ast.dump(tree, include_attributes=False)
    prefill, = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'prefill']
    decode, = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'decode']
    assert ast.unparse(prefill.body[-1]) == 'return (raw, token)'
    assert ast.unparse(decode.body[-1]) == 'return (raw, nxt, forward_ms, predicted)'
    prefill.body[-1:-1] = ast.parse('del out\nif use_cuda:\n    torch.cuda.empty_cache()').body
    decode.body[-1:-1] = ast.parse('del out, tup').body
    control_ast = ast.dump(tree, include_attributes=False)

    def compile_profile():
        namespace = vars(latency_profile).copy()
        exec(compile(ast.fix_missing_locations(tree), '<charged-decode-reclaim>', 'exec'), namespace)
        return namespace['profile_one']

    control = compile_profile()
    addition = ast.parse('if use_cuda:\n    torch.cuda.empty_cache()').body
    decode.body[-1:-1] = addition
    candidate_ast = ast.dump(tree, include_attributes=False)
    candidate = compile_profile()
    del decode.body[-2]
    assert ast.dump(tree, include_attributes=False) == control_ast
    del decode.body[-2]
    del prefill.body[-3:-1]
    assert ast.dump(tree, include_attributes=False) == original
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return control, candidate, dict(original_ast_sha256=digest(original),
        control_ast_sha256=digest(control_ast), candidate_ast_sha256=digest(candidate_ast),
        only_charged_decode_cache_release_differs=True)


def main():
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
    assert not torch.cuda.is_initialized()
    control, candidate, identity = profiles()
    previous = trial.profile_one, screen.cleanup_profile

    def observe(function):
        def profile(*args, **kwargs):
            result = function(*args, **kwargs)
            segments = [s for s in torch.cuda.memory_snapshot() if s['device'] == torch.cuda.current_device()]
            totals = dict(reserved_bytes=sum(s['total_size'] for s in segments),
                allocated_bytes=sum(s['allocated_size'] for s in segments),
                active_bytes=sum(s['active_size'] for s in segments),
                inactive_in_live_segments_bytes=sum(b['size'] for s in segments if s['active_size'] > 0
                                                   for b in s['blocks'] if b['state'] == 'inactive'),
                fully_inactive_segment_bytes=sum(s['total_size'] for s in segments if s['active_size'] == 0))
            assert totals['reserved_bytes'] == torch.cuda.memory_reserved()
            assert totals['allocated_bytes'] == torch.cuda.memory_allocated()
            assert totals['active_bytes'] + totals['inactive_in_live_segments_bytes'] + totals['fully_inactive_segment_bytes'] == totals['reserved_bytes']
            result['decode_reclaim'] = dict(setting=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
                snapshot_totals=totals, segments=segments,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations='Public final boundary snapshot cannot establish peak saving. Charged cleanup inside decode; original legacy arm names mean control/reclaim. Research caller only.')
            return result
        return profile

    trial.profile_one = observe(control)
    screen.cleanup_profile = lambda: (observe(candidate), identity)
    try: screen.main()
    finally: trial.profile_one, screen.cleanup_profile = previous


if __name__ == '__main__': main()
