"""Re-score saved predictions and verify paired prompts and sampled uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments'))
from longbench_metrics import DATASET_TO_METRIC, DATASET_TO_PROMPT
from longbench_eval import NO_CHAT_TASKS, task_data_path
from audit_unused_documents import document_group_key
from provenance import read_jsonl

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
a = p.parse_args()
output = a.directory/'verified_quality.json'
assert not output.exists(), 'Preserve evidence'
files = [a.directory/name for name in ['qa.json', 'retrieval_summary.json']]
docs = [json.loads(path.read_text()) for path in files]
assert docs[0]['config']['runtime'] == docs[1]['config']['runtime']
assert docs[0]['config']['model_identity'] == docs[1]['config']['model_identity']
results = {}
for doc in docs:
    assert not set(results) & set(doc['results'])
    results.update(doc['results'])
identities = {}; groups = {}; checked = 0
for task, arms in results.items():
    config = next(d['config'] for d in docs if task in d['results'])
    path = task_data_path(config['data_dir'], task)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config['dataset_sha256'][task]
    source = read_jsonl(path)[:config['max_samples']]
    identities[task] = [hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() for row in source]
    groups[task] = [document_group_key(row, task) for row in source]
    expected_prompts = []
    for row in source:
        prompt = DATASET_TO_PROMPT[task].format(**row)
        if task not in NO_CHAT_TASKS:
            prompt = f'[INST] {prompt} [/INST]'
        expected_prompts.append(hashlib.sha256(prompt.encode()).hexdigest())
    reference_tokens = None
    for method, entry in arms.items():
        assert entry['n'] == len(source) == len(entry['samples'])
        assert [r['sample_id'] for r in entry['samples']] == identities[task]
        assert [r['prompt_sha256'] for r in entry['samples']] == expected_prompts
        token_ids = [r['input_ids_sha256'] for r in entry['samples']]
        if reference_tokens is None:
            reference_tokens = token_ids
        assert token_ids == reference_tokens
        scores = []
        for raw, row in zip(entry['samples'], source):
            prediction = raw['prediction']
            if task in ('trec', 'triviaqa', 'samsum', 'lsht'):
                prediction = prediction.lstrip('\n').split('\n')[0]
            score = 100 * max(DATASET_TO_METRIC[task](prediction, ans, all_classes=row.get('all_classes', [])) for ans in row['answers'])
            assert abs(score - raw['score']) < 1e-9, (task, method)
            scores.append(score); checked += 1
        assert round(float(np.mean(scores)), 2) == entry['score']
    identities[task] = {'sample_ids': identities[task], 'token_ids_sha256': reference_tokens}
means = {method: float(np.mean([np.mean([row['score'] for row in results[task][method]['samples']]) for task in results])) for method in next(iter(results.values()))}
comparisons = {}
for method in ['tieredkv']:
    rng = np.random.default_rng(0)
    diff = {task: np.asarray([l['score']-r['score'] for l,r in zip(results[task][method]['samples'], results[task]['snapkv']['samples'])]) for task in results}
    boots = np.mean([values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1) for values in diff.values()], axis=0)
    comparisons[method] = {'mean_difference_vs_adapted_snap': float(np.mean([v.mean() for v in diff.values()])), 'row_paired_bootstrap_95': np.quantile(boots, [.025, .975]).tolist(), 'limits': 'Conditional sampled development rows; document dependence and one model/seed prevent an independent generalization claim.'}
report = {'verified': True, 'rescored_predictions': checked, 'task_balanced_means': means, 'comparisons': comparisons,
          'distinct_document_groups_per_task': {task: len(set(keys)) for task,keys in groups.items()},
          'source_runtime': docs[0]['config']['runtime'], 'paired_prompt_token_sample_identity': identities,
          'evidence_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
          'scope': 'Development pipeline reproduction only; adapted baselines, not faithful SOTA. Costs and baselines must be interpreted at their recorded source revision.', 'adopted': False}
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['paired_prompt_token_sample_identity', 'source_runtime', 'evidence_sha256']}, indent=2))
