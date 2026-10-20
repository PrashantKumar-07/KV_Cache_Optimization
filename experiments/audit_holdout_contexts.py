"""Audit exact document reuse in the frozen two-tranche evaluation."""
import hashlib
import json
from pathlib import Path
from provenance import atomic_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/baseline_fidelity/real_query_holdout_20260929'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    records = []
    manifests = {}
    earlier_contexts = {}
    for tranche, directory in enumerate((BASE, BASE / 'batch2'), 1):
        path = directory / 'manifest.json'
        manifests[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = json.loads(path.read_text())
        for task, meta in manifest['tasks'].items():
            source = ROOT / meta['source_file']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source_sha256']
            rows = read_jsonl(source)
            development = {digest(r['context']) for r in rows[:meta['excluded_prior_prefix_count']]}
            seen = earlier_contexts.setdefault(task, set())
            for index, sample_id in zip(meta['selected_zero_based_indices'], meta['selected_sample_ids']):
                row = rows[index]
                assert digest(json.dumps(row, sort_keys=True)) == sample_id
                context = digest(row['context'])
                records.append({'tranche': tranche, 'task': task, 'source_row': index,
                                'sample_id': sample_id, 'context_sha256': context,
                                'development_context_overlap': context in development,
                                'earlier_evaluation_context_overlap': context in seen})
                seen.add(context)
    tasks = {}
    for task in earlier_contexts:
        selected = [r for r in records if r['task'] == task]
        tasks[task] = {'rows': len(selected),
                      'unique_contexts': len({r['context_sha256'] for r in selected}),
                      'development_context_overlap_rows': sum(r['development_context_overlap'] for r in selected),
                      'reused_evaluation_context_rows': sum(r['earlier_evaluation_context_overlap'] for r in selected)}
    report = {'scope': 'Exact raw context SHA256 only; does not detect near-duplicate documents or model-training contamination.',
              'manifest_sha256': manifests, 'per_task': tasks, 'rows': records,
              'interpretation': 'Distinct sample IDs do not imply independent documents. Scores remain valid for these rows; row-bootstrap intervals do not account for shared-document dependence. Do not claim unseen-document generalization or calibrated replication significance.'}
    atomic_json(BASE / 'context_overlap_audit.json', report)
    print(json.dumps(tasks, indent=2))


if __name__ == '__main__':
    main()
