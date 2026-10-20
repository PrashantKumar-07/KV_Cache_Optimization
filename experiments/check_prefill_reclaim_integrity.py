"""Frozen generation and recycling gates for caller-only CUDA cache cleanup."""
import argparse
import ast
import hashlib
import inspect
from pathlib import Path
import sys
import torch
import longbench_eval as bench
import check_reuse_integrity as generation
import check_exact_pool_requests as requests
from exact_pinned_pool import ExactSharedPool, ExactPinnedPoolPolicy
from exact_pinned_memory import stats


class ExactControl(generation.Trajectory, ExactPinnedPoolPolicy):
    def __init__(self, **options):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)


class CleanupCandidate(generation.Candidate):
    direct_pool_cls = ExactSharedPool


def cleanup_generator():
    tree = ast.parse(inspect.getsource(bench.generate_with_policy))
    baseline = ast.dump(tree, include_attributes=False)
    function = tree.body[0]
    loops = [(i, n) for i, n in enumerate(function.body) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step']
    assert len(loops) == 1
    index = loops[0][0]
    additions = ast.parse("del out\nif str(device).startswith('cuda'):\n    torch.cuda.empty_cache()").body
    function.body[index:index] = additions
    changed = ast.dump(tree, include_attributes=False)
    del function.body[index:index+2]
    assert ast.dump(tree, include_attributes=False) == baseline
    function.body[index:index] = additions
    namespace = vars(bench).copy()
    exec(compile(ast.fix_missing_locations(tree), '<generation-prefill-reclaim>', 'exec'), namespace)
    return namespace['generate_with_policy'], dict(original_ast_sha256=hashlib.sha256(baseline.encode()).hexdigest(),
        candidate_ast_sha256=hashlib.sha256(changed.encode()).hexdigest(), only_predecode_cleanup_changed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('generation', 'interleaved'), required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--default-geometry', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.kind == 'generation' and args.manifest is None: parser.error('--manifest required')
    original_argv = sys.argv
    files = list(Path('src').glob('*.py')) + list(Path('experiments').glob('*.py')) + list(Path('experiments').glob('*.cu')) + list(Path('experiments').glob('*.cpp'))
    identity = dict(prefill_cleanup=True, cleanup_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if args.kind == 'generation':
        driver = generation
        compiled, ast_identity = cleanup_generator()
        identity.update(generation_cleanup_ast=ast_identity,
            arm_mapping={'fused_bounded': 'exact_reuse_control', 'native_reuse': 'exact_reuse_cleanup'})
        original_control, original_candidate, original_generate = generation.Control, generation.Candidate, bench.generate_with_policy

        def generate(model, tokenizer, input_text, maximum, policy, device, **options):
            function = compiled if isinstance(policy, CleanupCandidate) else original_generate
            return function(model, tokenizer, input_text, maximum, policy, device, **options)

        generation.Control, generation.Candidate, bench.generate_with_policy = ExactControl, CleanupCandidate, generate
        sys.argv = ['check', '--manifest', str(args.manifest), '--output', str(args.output)]
    else:
        import check_cpu_interleaved_model as driver
        original_start = driver.start

        def start(*positional, **options):
            state = original_start(*positional, **options)
            if options.get('fused_bounded'):
                torch.cuda.empty_cache()
            return state

        driver.start = start
        sys.argv = ['check', '--kind', 'interleaved', '--output', str(args.output)]
        if args.default_geometry: sys.argv.append('--default-geometry')
    original_write = driver.atomic_json

    def write(path, report):
        report.update(identity, exact_allocation_stats=stats())
        original_write(path, report)

    driver.atomic_json = write
    try:
        (generation.main if args.kind == 'generation' else requests.main)()
        assert stats()['live_allocations'] == 0, 'Exact root remains owned after reset'
    finally:
        driver.atomic_json, sys.argv = original_write, original_argv
        if args.kind == 'generation':
            generation.Control, generation.Candidate, bench.generate_with_policy = original_control, original_candidate, original_generate
        else:
            driver.start = original_start


if __name__ == '__main__':
    main()
