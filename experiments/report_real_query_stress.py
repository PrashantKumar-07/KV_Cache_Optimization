"""Inspect paired access-time placement failures after a significant gain."""
import hashlib
import json
from pathlib import Path

from provenance import atomic_json


ROOT = Path(__file__).resolve().parents[1] / 'results/baseline_fidelity'
PAIRS = (('real_q_lru_control_125.json', 'real_q_lru_candidate_125.json'),
         ('real_q_lru_control_gov10.json', 'real_q_lru_candidate_gov10.json'))


def bucket(length, boundaries):
    for boundary in boundaries:
        if length <= boundary:
            return f'<= {boundary}'
    return f'> {boundaries[-1]}'


def main():
    rows = []
    inputs = {}
    for left, right in PAIRS:
        paths = (ROOT / left, ROOT / right)
        inputs.update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        base, actual = [json.loads(p.read_text()) for p in paths]
        assert base['config']['tasks'] == actual['config']['tasks']
        for task in base['config']['tasks']:
            a = base['results'][task]['tieredkv']['samples']
            b = actual['results'][task]['tieredkv']['samples']
            for proxy, candidate in zip(a, b):
                assert proxy['sample_id'] == candidate['sample_id']
                assert proxy['prompt_tokens'] == candidate['prompt_tokens']
                rows.append({'task': task, 'sample_id': proxy['sample_id'],
                    'prompt_tokens': proxy['prompt_tokens'],
                    'proxy_generated_tokens': proxy['generated_tokens'],
                    'actual_generated_tokens': candidate['generated_tokens'],
                    'proxy_score': proxy['score'], 'actual_score': candidate['score'],
                    'difference': candidate['score'] - proxy['score']})
    bins = {}
    for label, field, boundaries in (('prompt_length', 'prompt_tokens', (4096, 16384)),
                                      ('proxy_generation_length', 'proxy_generated_tokens', (32, 128))):
        bins[label] = {}
        for row in rows:
            name = bucket(row[field], boundaries)
            bins[label].setdefault(name, []).append(row['difference'])
        bins[label] = {name: {'n': len(values),
                              'mean_difference': sum(values) / len(values),
                              'regressed': sum(value < 0 for value in values),
                              'severe_drop_gt_10': sum(value < -10 for value in values)}
                       for name, values in bins[label].items()}
    report = {'input_sha256': inputs, 'n': len(rows), 'bins': bins,
              'zero_to_positive': sum(r['proxy_score'] == 0 < r['actual_score'] for r in rows),
              'positive_to_zero': sum(r['actual_score'] == 0 < r['proxy_score'] for r in rows),
              'severe_drop_gt_10': sum(r['difference'] < -10 for r in rows),
              'worst_10': sorted(rows, key=lambda row: row['difference'])[:10],
              'limits': 'Selected LongBench examples only. No adversarial, out-of-distribution, '
                        'temporal-drift, multi-request, physical-tier or multi-model stress was run.'}
    atomic_json(ROOT / 'real_q_lru_stress_report.json', report)
    print(json.dumps({k: report[k] for k in ('n', 'bins', 'zero_to_positive',
                                             'positive_to_zero', 'severe_drop_gt_10')}, indent=2))


if __name__ == '__main__':
    main()
