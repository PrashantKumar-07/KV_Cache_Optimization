"""Frozen availability-conditional NarrativeQA placement and CPU parity study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import longbench_eval as bench
from check_batched_generation import measure
from provenance import atomic_json, runtime_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    manifest = json.loads(args.manifest.read_text())
    assert len(manifest['requests']) == 16 and len(manifest['excluded']) == 4
    sources = json.loads(args.manifest.with_name('source_manifest.json').read_text())
    for path, expected in sources['source_sha256'].items():
        assert sha(path) == expected, path
    report = dict(passed=False, pid=os.getpid(), runtime=runtime_identity(), rows=[],
        manifest_sha256=sha(args.manifest), source_sha256=sources['source_sha256'],
        expected_rows=48)
    try:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'])
        report['gpu'] = torch.cuda.get_device_name()
        for index, request in enumerate(manifest['requests']):
            path = Path(request['text_file'])
            assert sha(path) == request['text_sha256']
            item = dict(context=path.read_bytes().decode('utf-8'), input=request['question'],
                answers=request['answers'], all_classes=[])
            prompt = bench.DATASET_TO_PROMPT['narrativeqa'].format(**item)
            if 'narrativeqa' not in bench.NO_CHAT_TASKS:
                prompt = f'[INST] {prompt} [/INST]'
            ids = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids
            original_tokens = ids.size(1)
            if original_tokens > manifest['max_ctx']:
                half = manifest['max_ctx'] // 2
                ids = torch.cat((ids[:, :half], ids[:, -(manifest['max_ctx'] - half):]), dim=1)
            input_hash = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            pair = {}
            arms = manifest['arms']; offset = index % 3
            for arm in arms[offset:] + arms[:offset]:
                result = measure(model, tokenizer, item, 'narrativeqa', 0,
                    'gpu' if arm.startswith('gpu') else 'cpu_scores', manifest['max_ctx'],
                    prompt_queries=arm != 'gpu_proxy')
                assert result['generation']['prompt_tokens'] == ids.size(1)
                assert result['max_gen'] == 128
                shapes = result['prefill_query_shapes']
                if arm == 'gpu_proxy':
                    assert not shapes
                else:
                    assert len(shapes) == 32 and all(s == [32, 128, 128] for s in shapes.values())
                pair[arm] = result
                report['rows'].append(dict(request_index=index, story_id=request['story_id'],
                    kind=request['kind'], arm=arm, original_prompt_tokens=original_tokens,
                    removed_middle_tokens=original_tokens - ids.size(1), retained_input_ids_sha256=input_hash,
                    **result))
                atomic_json(args.output, report)
                print(index + 1, '/16', request['kind'], arm, len(result['generated_token_ids']), result['score'], flush=True)
            for field in ('prediction', 'generated_token_ids', 'score', 'generation', 'max_gen',
                'termination', 'prompt_sha256', 'state_sha256', 'trajectory_sha256',
                'trajectory_frames', 'counters', 'prefill_query_shapes'):
                assert pair['gpu_query'][field] == pair['cpu_query'][field], (request['story_id'], field)
        assert len(report['rows']) == 48
        report.update(passed=True, exact_cpu_gpu_actual_query_parity=True,
            limitations='Availability-conditional16/20 current adapted official NarrativeQA test stories, not LongBench published slice;10books/6movies,4source failures not replaced. Standard middle truncation, one model/seed. Instrumented timings descriptive only; quality requires separate preregistered paired bootstrap verifier. No population, speed, matched-baseline or physical STT claim.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
