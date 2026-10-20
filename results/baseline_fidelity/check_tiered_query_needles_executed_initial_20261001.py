"""Isolate existing actual-query prompt ranking at unchanged Tiered budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from check_batched_boundaries import measure
from provenance import atomic_json, runtime_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    sources = ['experiments/check_tiered_query_needles.py', 'experiments/check_batched_boundaries.py',
        'experiments/check_batched_generation.py', 'experiments/evaluate_query_selection.py',
        'experiments/compare_cpu_batched.py', 'experiments/longbench_eval.py', 'src/tiered_kv_cache.py']
    report = dict(passed=False, rows=[], runtime=runtime_identity(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    try:
        manifest = json.loads(args.manifest.read_text())
        cases = [case for case in manifest['cases'] if case.get('oracle')]
        prior_path = args.manifest.with_name('result.json')
        prior = json.loads(prior_path.read_text())
        report['prior_raw_sha256'] = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        controls = {row['case']: row for row in prior['rows'] if row['arm'] == 'gpu'}
        assert len(cases) == 6 and manifest['seed'] == 0
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'])
        report['gpu'] = torch.cuda.get_device_name()
        for number, case in enumerate(cases):
            assert hashlib.sha256(json.dumps(case['ids'], separators=(',', ':')).encode()).hexdigest() == case['token_sha256']
            pair = {}
            for arm in (('proxy', 'actual_query') if number % 2 == 0 else ('actual_query', 'proxy')):
                result = measure(model, tokenizer, case, 'gpu', prompt_queries=arm == 'actual_query')
                if arm == 'proxy':
                    for field in ('generated_token_ids', 'prediction', 'state_sha256', 'trajectory_sha256',
                                  'trajectory_frames', 'initial_answer_retention', 'counters'):
                        assert result[field] == controls[case['name']][field], field
                    assert not result['prefill_query_shapes']
                else:
                    assert len(result['prefill_query_shapes']) == 32
                    assert all(shape == [32, 128, 128] for shape in result['prefill_query_shapes'].values())
                pair[arm] = result
                report['rows'].append(dict(case=case['name'], arm=arm, token_sha256=case['token_sha256'], **result))
                atomic_json(args.output, report)
                print(case['name'], arm, result['answer_correct'], 'retained layers',
                    sum(x['retained_tokens'] == x['span_tokens'] for x in result['initial_answer_retention'].values()), flush=True)
            assert pair['proxy']['initial_cache_rows'] == pair['actual_query']['initial_cache_rows']
        report.update(passed=True, initial_cache_geometry_and_payload_equal=True,
            control_reproduces_frozen_boundary_run=True,
            correct_cases={arm: sum(row['answer_correct'] for row in report['rows'] if row['arm'] == arm)
                           for arm in ('proxy', 'actual_query')},
            limitations='Six reused synthetic development cases. Same initial budgets/storage and decode policy; final free-generation state and resources may differ. Existing real-query option remains rejected for broader factual/prefill gates. Instrumented timings/peaks are diagnostic only.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
