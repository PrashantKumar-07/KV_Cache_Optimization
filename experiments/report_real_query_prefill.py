"""Audit the paired actual-query prompt-placement stress pilot."""
import hashlib
import json
from pathlib import Path
import statistics

import longbench_eval as bench
from report_remediation import compare
from provenance import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/baseline_fidelity/real_q_prefill_control_qa5.json'
CAND = ROOT / 'results/baseline_fidelity/real_q_prefill_candidate_qa5.json'
OUT = ROOT / 'results/baseline_fidelity/real_q_prefill_report.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a, b = json.loads(BASE.read_text()), json.loads(CAND.read_text())
    ac, bc = a['config'], b['config']
    assert ac['real_query_prefill'] is False and bc['real_query_prefill'] is True
    assert {k: v for k, v in ac.items() if k != 'real_query_prefill'} == {
        k: v for k, v in bc.items() if k != 'real_query_prefill'}
    paired, measures = {}, {}
    for task in ac['tasks']:
        base = a['results'][task]['tieredkv']
        cand = b['results'][task]['tieredkv']
        data_path = bench.task_data_path(ac['data_dir'], task)
        assert sha(data_path) == ac['dataset_sha256'][task]
        with data_path.open(encoding='utf-8') as stream:
            source = [json.loads(line) for line in stream if line.strip()][:ac['max_samples']]
        for arm in (base, cand):
            assert arm['n'] == len(arm['samples']) == len(source)
            for row, item in zip(arm['samples'], source):
                expected_id = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                assert row['sample_id'] == expected_id
                expected_score = 100 * max(bench.DATASET_TO_METRIC[task](
                    row['prediction'], answer, all_classes=item.get('all_classes', []))
                    for answer in item['answers'])
                assert abs(expected_score - row['score']) < 1e-8
                assert 0 < row['final_resident_kv_payload_bytes'] <= 512 * 1024 * 1024
                assert row['prefill_s'] > 0 and row['prefill_peak_alloc_bytes'] > 0
        paired[task] = {'tieredkv': cand, 'proxy': base}
        measures[task] = {
            key: {name: statistics.median(row[field] for row in arm['samples'])
                  for name, arm in (('proxy', base), ('actual_query', cand))}
            for key, field in (('prefill_s_median', 'prefill_s'),
                               ('prefill_peak_alloc_bytes_median', 'prefill_peak_alloc_bytes'),
                               ('final_resident_kv_bytes_median', 'final_resident_kv_payload_bytes'))}
        measures[task]['paired_score_differences'] = [
            x['score'] - y['score'] for x, y in zip(cand['samples'], base['samples'])]
    report = {'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (BASE, CAND)},
              'report_script_sha256': sha(Path(__file__)),
              'runtime': ac['runtime'], 'quality': compare(paired, 'proxy', tasks=ac['tasks']),
              'measures': measures,
              'limitations': 'Five selected samples per task, one model/seed, no held-out claim. '
                             'Prefill time and peak are single-run descriptive measures; '
                             'decode TPOT, throughput, energy and physical STT are not measured.'}
    atomic_json(OUT, report)
    print(json.dumps({'quality': report['quality'], 'measures': measures}, indent=2))


if __name__ == '__main__':
    main()
