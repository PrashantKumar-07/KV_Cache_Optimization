"""Paired six-task comparison against the upstream-selector adapter.

Reports complete-policy differences, not equal-byte or speed comparisons.
"""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
from provenance import atomic_json
from report_remediation import compare, TASKS


def main():
    root = Path('results/snapkv_adapter_full')
    baseline, configs = {}, []
    for stem in ('short', 'gov'):
        path = root / f'{stem}.json'
        subprocess.run([sys.executable, str(Path(__file__).with_name('report_snapkv_adapter.py')),
                        str(path)], check=True, text=True)
        doc = json.loads(path.read_text())
        configs.append(doc['config'])
        assert not set(baseline) & set(doc['results'])
        baseline.update(doc['results'])
    variable = {'tasks', 'max_samples', 'dataset_sha256'}
    assert {k:v for k,v in configs[0].items() if k not in variable} == {
        k:v for k,v in configs[1].items() if k not in variable}
    datasets = {k:v for c in configs for k,v in c['dataset_sha256'].items()}
    assert set(baseline) == set(TASKS)
    comparisons = {}
    for directory in ('query_selection_full', 'query_selection_matched', 'query_selection_equal_total'):
        candidate = {}
        for stem in ('short', 'gov'):
            path = Path('results') / directory / f'query_{stem}.json'
            doc = json.loads(path.read_text())
            c = doc['config']
            selector = Path(__file__).with_name('evaluate_query_selection.py')
            assert c['selector_sha256'] == hashlib.sha256(selector.read_bytes()).hexdigest()
            if 'driver_sha256' in c:
                driver = Path(__file__).with_name('evaluate_query_budget.py')
                assert c['driver_sha256'] == hashlib.sha256(driver.read_bytes()).hexdigest()
            for key in ('model', 'data_dir', 'seed', 'max_ctx', 'runtime'):
                assert c[key] == configs[0][key], (directory, key)
            for task, digest in c['dataset_sha256'].items():
                assert datasets[task] == digest
            candidate.update(doc['results'])
        assert set(candidate) == set(TASKS)
        paired = {t:{'tieredkv':candidate[t]['tieredkv'],
                     'snapkv_upstream_selector':baseline[t]['snapkv_upstream_selector']} for t in TASKS}
        comparisons[directory] = compare(paired, 'snapkv_upstream_selector', tasks=TASKS)
    report = {'comparisons':comparisons,
        'scope':'Paired fixed samples/model/source; complete-policy comparisons only.',
        'limitations':['Adapter expands query heads and appends generated KV; capacities are not byte matched.',
                       'Adapter uses window 32 and SDPA decode; earlier policies use different selection and attention paths.',
                       'Intervals condition on the selected tasks/examples/model/seed; no SOTA or latency claim.']}
    atomic_json(root / 'comparison.json', report)
    print(json.dumps({k:v['task_balanced_mean_difference'] for k,v in comparisons.items()}, indent=2))


if __name__ == '__main__':
    main()
