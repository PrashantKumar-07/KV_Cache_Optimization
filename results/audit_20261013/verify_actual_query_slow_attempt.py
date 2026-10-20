"""Preregistered development quality screen, with document-cluster uncertainty."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments'))
import longbench_eval as bench
from audit_unused_documents import document_group_key
from provenance import read_jsonl

BASE = Path('results/audit_20261013/actual_query_development')
out = BASE/'verified_quality.json'
assert not out.exists(), 'Preserve prior evidence'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads((BASE/'plan.json').read_text())
control, candidate = [json.loads((BASE/f'{arm}.json').read_text()) for arm in ('control','candidate')]
a, b = control['config'], candidate['config']
assert a['real_query_prefill'] is False and b['real_query_prefill'] is True
assert {k:v for k,v in a.items() if k!='real_query_prefill'} == {k:v for k,v in b.items() if k!='real_query_prefill'}
with tarfile.open(BASE/'source.tar.gz') as archive:
    for path, expected in plan['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == expected
        assert sha(ROOT/path) == expected, ('Source changed during frozen screen', path)
assert set(control['results']) == set(candidate['results'])
rng = np.random.default_rng(0)
boots = []; tasks = {}; verified_rows = 0
for task in control['results']:
    source = bench.task_data_path(a['data_dir'], task)
    assert sha(source) == a['dataset_sha256'][task]
    items = read_jsonl(source)[:a['max_samples']]
    arms = {arm: data['results'][task]['tieredkv'] for arm,data in [('control',control),('candidate',candidate)]}
    assert all(entry['n'] == len(items) for entry in arms.values())
    diffs = []; grouped = {}
    for index, item in enumerate(items):
        left, right = [arms[arm]['samples'][index] for arm in ('control','candidate')]
        expected_id = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        for key in ('sample_id','prompt_sha256','input_ids_sha256','prompt_tokens'):
            assert left[key] == right[key], (task, key)
        assert left['sample_id'] == expected_id
        for row in (left, right):
            prediction = row['prediction']
            if task in ('trec','triviaqa','samsum','lsht'):
                prediction = prediction.lstrip('\n').split('\n')[0]
            score = 100*max(bench.DATASET_TO_METRIC[task](prediction, ans, all_classes=item.get('all_classes', [])) for ans in item['answers'])
            assert abs(score-row['score']) < 1e-9
            verified_rows += 1
        diff = right['score'] - left['score']; diffs.append(diff)
        grouped.setdefault(document_group_key(item,task), []).append(diff)
    clusters = list(grouped.values())
    sums = np.array([sum(values) for values in clusters]); sizes = np.array([len(values) for values in clusters])
    draw = rng.integers(0, len(clusters), (10000,len(clusters)))
    boots.append(sums[draw].sum(axis=1)/sizes[draw].sum(axis=1))
    task = str(task)
    tasks[task] = {'rows':len(items),'document_groups':len(clusters),'control_mean':float(np.mean([r['score'] for r in arms['control']['samples']])),
                   'candidate_mean':float(np.mean([r['score'] for r in arms['candidate']['samples']])), 'paired_difference':float(np.mean(diffs)),
                   'regressions':int(sum(value<0 for value in diffs)),
                   'median_prefill_peak_ratio_descriptive':float(np.median([r['prefill_peak_alloc_bytes'] for r in arms['candidate']['samples']])/np.median([r['prefill_peak_alloc_bytes'] for r in arms['control']['samples']])),
                   'median_prefill_time_ratio_descriptive':float(np.median([r['prefill_s'] for r in arms['candidate']['samples']])/np.median([r['prefill_s'] for r in arms['control']['samples']]))}
interval = np.quantile(np.mean(boots,axis=0), [.025,.975]).tolist()
report = {'verified':True,'rescored_predictions':verified_rows,'task_balanced_mean_difference':float(np.mean([r['paired_difference'] for r in tasks.values()])),
          'document_cluster_bootstrap_95':interval, 'quality_advancement_screen_passed':bool(interval[0]>0), 'per_task':tasks,
          'method':'10000 draws, seed0; resample target-document clusters within each task, preserve all paired rows of sampled cluster, equal-weight fixed-task means',
          'scope':'Previously consumed development requests, six fixed tasks, one model/seed. This advances only a quality development screen; memory/timing are descriptive parallel-GPU observations, no independent/generalized/SOTA result or adoption.',
          'adopted':False,'evidence_sha256':{p.name:sha(p) for p in [BASE/'control.json',BASE/'candidate.json',BASE/'plan.json',BASE/'source.tar.gz',Path(__file__)]}}
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['per_task','evidence_sha256']},indent=2))
