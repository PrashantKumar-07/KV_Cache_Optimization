"""Independently verify the frozen six-row grouped-retention pilot, CPU only.

Run from the repository root:
../venv_kvcache/bin/python results/audit_20261013/reviewer/verify_grouped_pilot.py --output results/audit_20261013/reviewer/grouped_pilot_verified.json

Uses archived scoring code, not current implementation code. A new output
path is required on every rerun; prior evidence is never overwritten.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'results/audit_20261013/grouped_pilot'
SOURCE_SUFFIXES = {'.py', '.cpp', '.cu', '.h', '.hpp', '.sh', '.json'}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def read_archive(path):
    # Sequential access avoids repeated gzip decompression for every member.
    contents = {}
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            if member.isfile():
                assert member.name not in contents, member.name
                contents[member.name] = archive.extractfile(member).read()
    return contents


def verify():
    plan = json.loads((BASE / 'plan.json').read_text())
    docs = {arm: json.loads((BASE / f'{arm}.json').read_text())
            for arm in ('control', 'candidate')}
    archive = read_archive(BASE / 'source.tar.gz')
    expected = plan['source_sha256']
    assert set(expected) <= set(archive)
    assert all(sha(archive[name]) == digest for name, digest in expected.items())
    runtime = hashlib.sha256()
    # Frozen provenance.py sorts Path objects, which compare path components.
    for name in sorted(expected, key=PurePosixPath):
        path = PurePosixPath(name)
        if path.parts[0] in ('src', 'experiments', 'sota') and path.suffix in SOURCE_SUFFIXES:
            runtime.update(name.encode())
            runtime.update(archive[name])
    for doc in docs.values():
        assert runtime.hexdigest() == doc['config']['runtime']['source_sha256']
        assert sha(archive['experiments/grouped_retention.py']) == doc['config']['grouped_policy_sha256']
        assert sha(archive['experiments/evaluate_query_selection.py']) == doc['config']['selector_sha256']
        assert sha((BASE / 'run_arm.py').read_bytes()) == doc['config']['pilot_runner_sha256']
    historical = json.loads((Path(__file__).parent / 'grouped_source_before_guards_v2.json').read_text())
    assert historical['passed']
    assert historical['plan_sha256'] == sha((BASE / 'plan.json').read_bytes())
    assert historical['source_archive_sha256'] == sha((BASE / 'source.tar.gz').read_bytes())
    assert not historical['current_worktree_hash_mismatches_before_guard_edit']

    configs = {arm: doc['config'] for arm, doc in docs.items()}
    assert configs['control']['head_grouped_retention'] is False
    assert configs['candidate']['head_grouped_retention'] is True
    assert {k: v for k, v in configs['control'].items() if k != 'head_grouped_retention'} == {
        k: v for k, v in configs['candidate'].items() if k != 'head_grouped_retention'}
    config = configs['control']
    assert config['tasks'] == plan['tasks']
    assert config['max_samples'] == plan['samples_per_task'] == 2
    settings = plan['settings']
    mapping = {'hot': 'tiered_vram_budget', 'cold': 'tiered_stt_budget',
               'exposed': 'stt_expose_quota', 'recent': 'recent',
               'frac': 'sttram_bifurcation_frac', 'dtype': 'kv_dtype',
               'seed': 'seed', 'max_ctx': 'max_ctx', 'last_logits': 'prefill_last_logits'}
    assert all(config[mapping[key]] == value for key, value in settings.items())
    assert config['real_query_prefill'] and config['strategy'] == 'lru'
    assert not config['cpu_batched_scores']
    for package in ('rouge', 'transformers'):
        assert importlib.metadata.version(package) == config['runtime']['packages'][package]

    namespace = {'__name__': 'independent_frozen_grouped_metrics'}
    exec(compile(archive['experiments/longbench_metrics.py'],
                 'frozen/experiments/longbench_metrics.py', 'exec'), namespace)
    evaluator = ast.parse(archive['experiments/longbench_eval.py'])
    exclusions = next(ast.literal_eval(node.value) for node in evaluator.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id == 'NO_CHAT_TASKS' for t in node.targets))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str((ROOT / config['model']).resolve()),
                                              local_files_only=True)
    rows, per_task, datasets = [], {}, {}
    for task in plan['tasks']:
        path = ROOT / config['data_dir'] / f'{task}_e.jsonl'
        if not path.is_file():
            path = path.with_name(f'{task}.jsonl')
        assert sha(path.read_bytes()) == config['dataset_sha256'][task]
        data = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
        datasets[task] = dict(source_file=str(path), source_sha256=sha(path.read_bytes()))
        entries = {arm: docs[arm]['results'][task]['tieredkv'] for arm in docs}
        assert all(set(doc['results']) == set(plan['tasks']) for doc in docs.values())
        assert all(entry['n'] == len(entry['samples']) == 2 for entry in entries.values())
        task_scores = {arm: [] for arm in docs}
        differences = []
        for index, source in enumerate(data[:2]):
            samples = {arm: entries[arm]['samples'][index] for arm in docs}
            sample_id = sha(json.dumps(source, sort_keys=True).encode())
            prompt = namespace['DATASET_TO_PROMPT'][task].format(**source)
            if 'instruct' in config['model'].lower() and task not in exclusions:
                prompt = f'[INST] {prompt} [/INST]'
            ids = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids
            if ids.shape[1] > config['max_ctx']:
                half = config['max_ctx'] // 2
                ids = __import__('torch').cat((ids[:, :half], ids[:, -(config['max_ctx'] - half):]), 1)
            identity = dict(sample_id=sample_id, prompt_sha256=sha(prompt.encode()),
                input_ids_sha256=sha(ids.numpy().tobytes()), prompt_tokens=ids.shape[1],
                max_ctx=config['max_ctx'])
            scores = {}
            for arm, sample in samples.items():
                assert all(sample[key] == value for key, value in identity.items())
                prediction = sample['prediction']
                if task in ('trec', 'triviaqa', 'samsum', 'lsht'):
                    prediction = prediction.lstrip('\n').split('\n')[0]
                score = max(namespace['DATASET_TO_METRIC'][task](prediction, answer,
                    all_classes=source.get('all_classes', [])) for answer in source['answers']) * 100
                assert abs(score - sample['score']) < 1e-9
                assert round(score, 2) == entries[arm]['sample_scores'][index]
                task_scores[arm].append(score)
                scores[arm] = score
            difference = scores['candidate'] - scores['control']
            differences.append(difference)
            rows.append(dict(task=task, source_index=index, **identity,
                scores=scores, candidate_minus_control=difference,
                predictions={arm: sample['prediction'] for arm, sample in samples.items()}))
        means = {arm: sum(values) / len(values) for arm, values in task_scores.items()}
        assert all(round(means[arm], 2) == entries[arm]['score'] for arm in docs)
        per_task[task] = dict(means=means, candidate_minus_control=sum(differences) / len(differences))

    gain = sum(value['candidate_minus_control'] for value in per_task.values()) / len(per_task)
    primary = gain >= 3
    regression = all(value['candidate_minus_control'] >= -10 for value in per_task.values())
    return dict(verification_passed=True, recorded_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__).read_bytes()),
        input_sha256={name: sha((BASE / name).read_bytes())
                      for name in ('plan.json', 'source.tar.gz', 'run_arm.py', 'control.json', 'candidate.json')},
        frozen_source_files_checked=len(expected), frozen_runtime_source_sha256=runtime.hexdigest(),
        before_guard_worktree_proof_sha256=sha((Path(__file__).parent / 'grouped_source_before_guards_v2.json').read_bytes()),
        config_difference_only='head_grouped_retention', rows=rows, per_task=per_task,
        dataset_sources=datasets, independent_cpu_retokenization_passed=True,
        paired_rows_verified=len(rows), task_balanced_candidate_minus_control=gain,
        registered_gate=dict(primary_gain_at_least_3=primary,
                             no_task_mean_loss_over_10=regression,
                             advance=primary and regression),
        decision='advance' if primary and regression else 'reject; stop candidate expansion',
        scope='Six consumed development rows, two per QA task, one model/seed. Falsification pilot only.',
        limitations=['No independent quality/significance or novelty claim.',
            'Same KV payload capacities; total memory, metadata, transient repacking and actual final residency differ.',
            'Different outputs/lengths and simultaneous GPU arms prevent causal timing or write-rate conclusions.',
            'Raw runner records model path but not model artifact hashes; exact historical weight bytes are not independently attested here.',
            'Current implementation may contain later malformed-input guards; verification uses frozen pilot source and recorded before-edit proof.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), f'Preserve prior proof: {args.output}'
    report = verify()
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('verification_passed', 'paired_rows_verified',
        'task_balanced_candidate_minus_control', 'registered_gate', 'decision')}, indent=2))


if __name__ == '__main__':
    main()
