"""Verify CPU transfer/score batching preserves frozen actual-query generation."""
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
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    sources = ['experiments/check_cpu_query_needles.py', 'experiments/check_batched_boundaries.py',
        'experiments/check_batched_generation.py', 'experiments/evaluate_query_selection.py',
        'experiments/compare_cpu_batched.py', 'experiments/batched_score_policy.py',
        'experiments/cpu_batched_cache.py', 'experiments/cpu_slot_cache.py']
    report = dict(passed=False, rows=[], runtime=runtime_identity(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    try:
        manifest=json.loads(args.manifest.read_text()); ref=json.loads(args.reference.read_text())
        assert ref['passed'] and ref['manifest_sha256'] == report['manifest_sha256']
        reference={row['case']:row for row in ref['rows'] if row['arm']=='actual_query'}
        cases=[case for case in manifest['cases'] if case.get('oracle')]
        assert len(cases)==len(reference)==6 and manifest['seed']==0
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(manifest['model'])
        report['gpu']=torch.cuda.get_device_name()
        for case in cases:
            assert hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest()==case['token_sha256']
            result=measure(model,tokenizer,case,'cpu_scores',prompt_queries=True)
            for field in ('generated_token_ids','prediction','answer_correct','state_sha256','trajectory_sha256',
                          'trajectory_frames','initial_answer_retention','initial_cache_rows','prefill_query_shapes',
                          'counters','max_cold_rows_per_layer','largest_step_promoted','largest_step_paid_writes'):
                assert json.loads(json.dumps(result[field]))==reference[case['name']][field], field
            assert result['reserved_host_tensor_bytes']==402784256
            report['rows'].append(dict(case=case['name'],token_sha256=case['token_sha256'],**result))
            atomic_json(args.output,report)
            print(case['name'],result['answer_correct'],'exact GPU reference',flush=True)
        report.update(passed=True,exact_gpu_reference=True,correct_cases=sum(row['answer_correct'] for row in report['rows']),
            limitations='Six reused synthetic probes; physical CPU reservation/transfers differ. No independent quality, latency or matched-total-resource gate; defaults unchanged.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output,report)


if __name__=='__main__':
    main()
