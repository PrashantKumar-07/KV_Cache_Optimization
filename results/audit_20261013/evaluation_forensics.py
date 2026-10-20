"""CPU-only immutable forensic record; never selects or evaluates new samples."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from longbench_metrics import DATASET_TO_METRIC, DATASET_TO_PROMPT
from provenance import read_jsonl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(text.casefold().split())


def context_key(row):
    return hashlib.sha256(normalized(row['context']).encode()).hexdigest()


def target_passage(row):
    return normalized(row['input'].rsplit('\nQuestion:', 1)[0])


def main():
    target = Path(__file__).with_name('evaluation_forensics.json')
    if target.exists():
        raise FileExistsError(target)
    base = ROOT / 'results/baseline_fidelity'
    reference = json.loads((ROOT / 'sota/snapkv/experiments/LongBench/config/dataset2prompt.json').read_text())
    report = {'scope': 'CPU-only audit of frozen artifacts; no new selection or model evaluation',
              'script_sha256': sha(__file__), 'source_sha256': {
                  name: sha(ROOT / name) for name in ('experiments/longbench_metrics.py',
                      'experiments/longbench_eval.py', 'experiments/provenance.py',
                      'experiments/audit_unused_documents.py')},
              'prompt_differences': {task: {'local': prompt, 'reference': reference[task]}
                  for task, prompt in DATASET_TO_PROMPT.items() if prompt != reference[task]},
              'default_six_task_prompt_differences': ['qasper', 'narrativeqa']}
    first = json.loads((base / 'real_query_holdout_20260929/manifest.json').read_text())
    source = ROOT / first['tasks']['triviaqa']['source_file']
    trivia = read_jsonl(source)
    samsum = read_jsonl(source.with_name('samsum_e.jsonl'))[0]
    report['samsum_missing_target'] = {
        'source_file': str(source.with_name('samsum_e.jsonl')),
        'sample_id': hashlib.sha256(json.dumps(samsum, sort_keys=True).encode()).hexdigest(),
        'target_input_present_in_local_prompt': samsum['input'] in DATASET_TO_PROMPT['samsum'].format(**samsum),
        'target_input_present_in_reference_prompt': samsum['input'] in reference['samsum'].format(**samsum),
        'scope': 'Supported optional task; absent from six-task headline'}
    report['triviaqa_identity'] = {'source_sha256': sha(source), 'rows': len(trivia),
        'distinct_contexts': len({normalized(row['context']) for row in trivia}),
        'distinct_target_inputs': len({normalized(row['input']) for row in trivia}),
        'historical_holdout_overlap': {}}
    for name in ('realq_cpu_fresh_documents_20261001', 'snap_tier_envelope_fresh_quality_20261002'):
        manifest_path = base / name / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        inventory_path = ROOT / manifest.get('inventory_file', 'results/baseline_fidelity/unused_document_inventory_20261001.json')
        inventory = json.loads(inventory_path.read_text())
        used = set(inventory['excluded_normalized_context_keys'])
        used_targets = {}
        for index, row in enumerate(trivia):
            if context_key(row) in used:
                used_targets.setdefault(target_passage(row), index)
        selected = manifest['tasks']['triviaqa']['selected_zero_based_indices']
        report['triviaqa_identity']['historical_holdout_overlap'][name] = {
            'manifest_sha256': sha(manifest_path), 'inventory_sha256': sha(inventory_path),
            'selected_rows': len(selected), 'already_used_target_passages': [
                {'selected_source_index': i, 'prior_source_index': used_targets[target_passage(trivia[i])]}
                for i in selected if target_passage(trivia[i]) in used_targets]}
    inventory_path = base / 'unused_document_inventory_20261003.json'
    inventory = json.loads(inventory_path.read_text())
    eligible = [i for indices in inventory['tasks']['triviaqa']['eligible_groups'].values() for i in indices]
    excluded = set(inventory['excluded_normalized_context_keys'])
    used = {target_passage(row) for row in trivia if context_key(row) in excluded}
    report['triviaqa_identity']['readiness_20261003'] = {
        'inventory_sha256': sha(inventory_path), 'claimed_eligible_context_groups': len(eligible),
        'distinct_target_passages': len({target_passage(trivia[i]) for i in eligible}),
        'eligible_rows_with_already_used_targets': sum(target_passage(trivia[i]) in used for i in eligible),
        'unconsumed_target_passages_in_e_source': len({target_passage(trivia[i]) for i in eligible} - used),
        'scope': 'Reinterprets existing context exclusion inventory in its E source; not new cross-variant selection'}
    directory = base / 'snap_tier_envelope_fresh_quality_20261002'
    manifest = json.loads((directory / 'manifest.json').read_text())
    raw = json.loads((directory / 'result.json').read_text())
    verified = json.loads((directory / 'verified_report.json').read_text())
    diffs, checked = [], 0
    for task, meta in manifest['tasks'].items():
        source = ROOT / meta['slice_file']
        assert sha(source) == meta['slice_sha256']
        differences = []
        for item in read_jsonl(source):
            sample = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            pair = {row['arm']: row for row in raw['rows'] if row['task'] == task and row['sample_id'] == sample}
            assert set(pair) == {'snap', 'cpu_direct'}
            for row in pair.values():
                prediction = row['prediction']
                if task in ('trec', 'triviaqa', 'samsum', 'lsht'):
                    prediction = prediction.lstrip('\n').split('\n')[0]
                score = 100 * max(DATASET_TO_METRIC[task](prediction, answer, all_classes=item.get('all_classes', [])) for answer in item['answers'])
                assert abs(score - row['score']) < 1e-8
                checked += 1
            differences.append(pair['cpu_direct']['score'] - pair['snap']['score'])
        diffs.append(np.asarray(differences))
    rng = np.random.default_rng(manifest['bootstrap_seed'])
    draws = np.mean(np.stack([rng.choice(d, size=(manifest['bootstrap_draws'], len(d)), replace=True).mean(1) for d in diffs]), axis=0)
    mean = float(np.mean([d.mean() for d in diffs]))
    ci = np.quantile(draws, [.025, .975]).tolist()
    assert mean == verified['quality']['task_balanced_mean_difference']
    assert ci == verified['quality']['paired_document_bootstrap_ci95']
    assert sha(directory / 'result.json') == verified['raw_sha256']
    report['independent_rescore'] = {'rows': checked, 'raw_sha256': sha(directory / 'result.json'),
        'task_balanced_difference': mean, 'historical_row_bootstrap_ci95_reproduced': ci,
        'interpretation': 'Stored scores and numerical interval reproduce exactly; target leakage above invalidates the unused-document interpretation, not these arithmetic checks.'}
    with target.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(target)


if __name__ == '__main__':
    main()
