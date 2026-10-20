"""Charged prefill cleanup with unchanged exact-root policy in both arms."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import latency_profile
import compare_cpu_batched as trial
import audit_exact_pool_resources as resources
from check_exact_pool_requests import RecordedExactPolicy
from exact_pinned_memory import stats
from provenance import atomic_json, runtime_identity


def cleanup_profile():
    tree = ast.parse(inspect.getsource(latency_profile.profile_one))
    baseline = ast.dump(tree, include_attributes=False)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'prefill']
    assert len(functions) == 1
    prefill = functions[0]
    assert ast.dump(prefill.body[-1], include_attributes=False) == ast.dump(ast.parse('return raw, token').body[0], include_attributes=False)
    additions = ast.parse('del out\nif use_cuda:\n    torch.cuda.empty_cache()').body
    prefill.body[-1:-1] = additions
    changed = ast.dump(tree, include_attributes=False)
    del prefill.body[-3:-1]
    assert ast.dump(tree, include_attributes=False) == baseline, 'Changed more than prefill cleanup'
    prefill.body[-1:-1] = additions
    namespace = vars(latency_profile).copy()
    exec(compile(ast.fix_missing_locations(tree), '<charged-prefill-reclaim>', 'exec'), namespace)
    return namespace['profile_one'], dict(
        original_ast_sha256=hashlib.sha256(baseline.encode()).hexdigest(),
        candidate_ast_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        only_charged_prefill_cleanup_changed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hot', type=int, default=991)
    parser.add_argument('--cold', type=int, default=1090)
    parser.add_argument('--decode', type=int, default=5)
    parser.add_argument('--blocks', type=int, choices=(0, 3), default=0)
    parser.add_argument('--warmup', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    assert (args.hot, args.cold) in ((991, 1090), (1024, 2048))
    candidate, identity = cleanup_profile()
    original_profile, original_main, original_policy = trial.profile_one, resources.comparison.main, trial.FusedBoundedPolicy
    arm = 'control'

    def dispatch(*positional, **options):
        return (original_profile if arm == 'control' else candidate)(*positional, **options)

    def run():
        nonlocal arm
        sources = list(Path('src').glob('*.py')) + list(Path('experiments').glob('*.py')) + list(Path('experiments').glob('*.cu')) + list(Path('experiments').glob('*.cpp'))
        report = dict(passed=False, runtime=runtime_identity(), rows=[],
            config=vars(args) | {'output': str(args.output), 'context': 16384},
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, **identity)
        try:
            prompt_path = Path('results/hardware/batched_eviction_coordinator_20261001/max_context_prompt.txt')
            teacher_path = Path('results/shadow_ablation_full/fixed_stream_20260929.json')
            prompt = prompt_path.read_text()
            stream = json.loads(teacher_path.read_text())['config']['teacher_token_ids']
            report.update(prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                teacher_source_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(), teacher_token_ids=stream[:args.decode+1])
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct', torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
            tokenizer = AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
            reference = None
            for block in range(args.blocks or 1):
                for arm in (('control', 'cleanup', 'cleanup', 'control') if args.blocks else ('control', 'cleanup')):
                    row = trial.measure(model, tokenizer, prompt, stream, 16384, args.decode, 'fused_bounded_scores',
                        args.warmup, args.blocks == 0, real_query_prefill=True, hot=args.hot, cold=args.cold)
                    if reference is None: reference = row
                    else:
                        trial.check(reference, row)
                        for field in ('reserved_host_bytes', 'native_d2h_calls', 'native_h2d_calls', 'per_layer_initial_d2h_calls'):
                            assert row[field] == reference[field], field
                    assert stats()['live_allocations'] == 0, 'Exact root remains owned after reset'
                    report['rows'].append(dict(arm=arm, block=block, **row))
                    report['exact_allocation_stats'] = stats()
                    atomic_json(args.output, report)
                    print(block, arm, 'exact', flush=True)
            report.update(passed=True, limitations='Same exact-root policy and frozen teacher in both arms; cleanup charged within TTFT. Pilot is direction-only; full ABBA/stress/resource scopes required. No quality superiority, physical STT, model-wide memory or serving claim.')
        except BaseException as error:
            report['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            atomic_json(args.output, report)

    trial.profile_one, resources.comparison.main, trial.FusedBoundedPolicy = dispatch, run, RecordedExactPolicy
    try:
        resources.main()
    finally:
        trial.profile_one, resources.comparison.main, trial.FusedBoundedPolicy = original_profile, original_main, original_policy


if __name__ == '__main__':
    main()
