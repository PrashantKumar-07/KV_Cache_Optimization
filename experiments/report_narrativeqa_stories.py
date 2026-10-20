"""Independent identity, generation parity and conditional story quality gates."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from transformers import AutoTokenizer
import longbench_eval as bench
from provenance import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    m = json.loads(args.manifest.read_text()); r = json.loads(args.raw.read_text())
    assert r['passed'] and len(r['rows']) == 48
    assert sha(args.manifest) == r['manifest_sha256']
    for file_key, hash_key in (('source_selection_manifest','source_selection_sha256'),
        ('preflight_file','preflight_sha256'), ('source_verification_file','source_verification_sha256')):
        assert sha(m[file_key]) == m[hash_key]
    selected = json.loads(Path(m['source_selection_manifest']).read_text())['selected']
    selected = {x['story']['document_id']:x for x in selected}
    preflight = {x['story_id']:x for x in json.loads(Path(m['preflight_file']).read_text())['rows']}
    assert {x['story_id'] for x in m['requests']} == {k for k,v in preflight.items() if v['usable']}
    assert {x['story_id'] for x in m['excluded']} == {k for k,v in preflight.items() if not v['usable']}
    source = json.loads(args.manifest.with_name('source_manifest.json').read_text())
    for path in ('experiments/longbench_eval.py', 'experiments/longbench_metrics.py'):
        assert sha(path) == source['source_sha256'][path]
    assert sha(args.manifest.with_name('source.tar.gz')) == source['source_bundle_sha256']
    with tarfile.open(args.manifest.with_name('source.tar.gz')) as archive:
        assert r['source_sha256'] == source['source_sha256']
        for path, expected in source['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == expected
        runtime_files = sorted(p for p in source['source_sha256'] if p.startswith('src/')) + [
            f'experiments/{p}' for p in ('longbench_eval.py','longbench_metrics.py','latency_profile.py','ppl_eval.py','provenance.py')]
        digest = hashlib.sha256()
        for path in runtime_files:
            digest.update(path.encode()); digest.update(archive.extractfile(path).read())
        assert digest.hexdigest() == r['runtime']['source_sha256']
    tokenizer = AutoTokenizer.from_pretrained(m['model'])
    differences = []; details = []; expected_order = []; seen = set()
    for index, request in enumerate(m['requests']):
        story = request['story_id']; assert story not in seen; seen.add(story)
        original = selected[story]; qa = original['qa']; row = preflight[story]
        assert original['story']['set'] == qa['set'] == 'test'
        assert request['kind'] == row['kind'] == original['story']['kind']
        assert request['question'] == qa['question'] and request['answers'] == [qa['answer1'],qa['answer2']]
        assert request['text_file'] == row['text_file'] and request['text_sha256'] == row['text_sha256']
        assert sha(request['text_file']) == request['text_sha256'] and sha(request['raw_file']) == request['raw_sha256']
        item = dict(context=Path(request['text_file']).read_bytes().decode('utf-8'), input=request['question'], answers=request['answers'])
        prompt = bench.DATASET_TO_PROMPT['narrativeqa'].format(**item)
        if 'narrativeqa' not in bench.NO_CHAT_TASKS: prompt = f'[INST] {prompt} [/INST]'
        ids = tokenizer(prompt, truncation=False, return_tensors='np')['input_ids']
        original_tokens = ids.shape[1]
        if original_tokens > 31500: ids = np.concatenate((ids[:,:15750],ids[:,-15750:]),axis=1)
        rows = [x for x in r['rows'] if x['story_id'] == story]; assert len(rows) == 3
        arms = {x['arm']:x for x in rows}; assert set(arms) == set(m['arms'])
        offset = index % 3
        expected_order.extend((story,arm) for arm in m['arms'][offset:] + m['arms'][:offset])
        for arm, result in arms.items():
            assert result['request_index'] == index and result['kind'] == request['kind']
            assert result['original_prompt_tokens'] == original_tokens
            assert result['removed_middle_tokens'] == original_tokens - ids.shape[1]
            assert result['retained_input_ids_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest()
            assert result['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            generated = result['generated_token_ids']; assert 1 <= len(generated) <= 128
            assert result['trajectory_frames'] == len(generated)
            assert result['counters']['decode_steps'] == len(generated) - 1
            assert tokenizer.decode(generated, skip_special_tokens=True) == result['prediction']
            assert result['generation'] == dict(prompt_tokens=ids.shape[1],generated_tokens=len(generated))
            assert result['max_gen'] == 128
            assert result['termination'] == ('eos' if generated[-1] == tokenizer.eos_token_id else 'cap')
            assert generated[-1] == tokenizer.eos_token_id or len(generated) == 128
            score = 100 * max(bench.DATASET_TO_METRIC['narrativeqa'](result['prediction'],answer,all_classes=[]) for answer in request['answers'])
            assert abs(score - result['score']) < 1e-10
            shapes = result['prefill_query_shapes']
            assert (not shapes) if arm == 'gpu_proxy' else (set(shapes) == {str(i) for i in range(32)} and all(s == [32,128,128] for s in shapes.values()))
        for field in ('prediction','generated_token_ids','score','generation','termination','prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes'):
            assert arms['gpu_query'][field] == arms['cpu_query'][field], (story,field)
        delta = arms['cpu_query']['score'] - arms['gpu_proxy']['score']; differences.append(delta)
        details.append(dict(story_id=story,kind=request['kind'],proxy_score=arms['gpu_proxy']['score'],candidate_score=arms['cpu_query']['score'],delta=delta,original_prompt_tokens=original_tokens,retained_prompt_tokens=ids.shape[1]))
    assert [(x['story_id'],x['arm']) for x in r['rows']] == expected_order
    values = np.array(differences); rng = np.random.default_rng(m['bootstrap_seed'])
    draws = values[rng.integers(0,len(values),size=(m['bootstrap_draws'],len(values)))].mean(axis=1)
    lower, upper = np.quantile(draws,[.025,.975])
    report = dict(passed=True,manifest_sha256=sha(args.manifest),raw_sha256=sha(args.raw),
        reporter_sha256=sha(__file__),verified_story_identity_truncation_scores_order_and_cpu_gpu_parity=True,
        selected_stories=20,evaluated_stories=16,source_failures=m['excluded'],
        mean_delta=float(values.mean()),paired_story_ci95=[float(lower),float(upper)],
        proxy_mean=float(np.mean([x['proxy_score'] for x in details])),
        candidate_mean=float(np.mean([x['candidate_score'] for x in details])),
        quality_noninferiority_passed=bool(lower > -.5),quality_superiority_passed=bool(lower > 0),
        regression_count=int((values < 0).sum()),loss_over_10_count=int((values < -10).sum()),
        kind_mean_deltas={kind:float(np.mean([x['delta'] for x in details if x['kind']==kind])) for kind in ('gutenberg','movie')},
        stories=details,limitations=m['limitations']+' One model/seed;16 availability-conditional paired stories; instrumented runtime is not speed evidence; no tuning after this holdout.')
    atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('stories','source_failures')},indent=2))


if __name__ == '__main__':
    main()
