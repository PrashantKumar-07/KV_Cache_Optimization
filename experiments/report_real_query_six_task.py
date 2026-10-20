"""Verify paired actual-query prompt-placement runs across six tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import longbench_eval as bench
from provenance import atomic_json
from report_remediation import compare

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/baseline_fidelity'
PILOT_PAIRS = [('real_q_prefill_control_qa5.json', 'real_q_prefill_candidate_qa5.json'),
         ('real_q_prefill_control_short5.json', 'real_q_prefill_candidate_short5.json'),
         ('real_q_prefill_control_gov2.json', 'real_q_prefill_candidate_gov2.json')]
FULL_PAIRS = [('real_q_full_control_qn25.json', 'real_q_full_candidate_qn25.json'),
              ('real_q_full_control_other25.json', 'real_q_full_candidate_other25.json'),
              ('real_q_full_control_gov10.json', 'real_q_full_candidate_gov10.json')]
LRU_FULL_PAIRS = [('real_q_lru_control_125.json', 'real_q_lru_candidate_125.json'),
                  ('real_q_lru_control_gov10.json', 'real_q_lru_candidate_gov10.json')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core(config):
    return {k: v for k, v in config.items()
            if k not in ('tasks', 'max_samples', 'dataset_sha256', 'real_query_prefill')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--lru-full', action='store_true')
    args = parser.parse_args()
    if args.full and args.lru_full:
        parser.error('--full and --lru-full are mutually exclusive')
    pairs = LRU_FULL_PAIRS if args.lru_full else FULL_PAIRS if args.full else PILOT_PAIRS
    output = DIR / ('real_q_lru_full_report.json' if args.lru_full else
                    'real_q_prefill_full_report.json' if args.full else
                    'real_q_prefill_six_task_report.json')
    refs, paired, measures = {}, {}, {}
    canonical = None
    for base_name, cand_name in pairs:
        paths = (DIR / base_name, DIR / cand_name)
        base, cand = [json.loads(p.read_text()) for p in paths]
        refs.update({p.name: sha(p) for p in paths})
        ac, bc = base['config'], cand['config']
        assert ac['real_query_prefill'] is False and bc['real_query_prefill'] is True
        assert core(ac) == core(bc)
        assert ac['strategy'] == ('lru' if args.lru_full else 'query')
        assert ac['tasks'] == bc['tasks'] and ac['max_samples'] == bc['max_samples']
        assert ac['dataset_sha256'] == bc['dataset_sha256']
        if canonical is None:
            canonical = core(ac)
        else:
            assert core(ac) == canonical
        for task in ac['tasks']:
            assert task not in paired
            a, b = base['results'][task]['tieredkv'], cand['results'][task]['tieredkv']
            data = bench.task_data_path(ac['data_dir'], task)
            assert sha(data) == ac['dataset_sha256'][task]
            with data.open(encoding='utf-8') as stream:
                items = [json.loads(line) for line in stream if line.strip()][:ac['max_samples']]
            for arm in (a, b):
                assert arm['n'] == len(arm['samples']) == len(items)
                assert arm['sample_scores'] == [round(row['score'], 2)
                                                for row in arm['samples']]
                assert abs(sum(row['score'] for row in arm['samples']) / len(items)
                           - arm['score']) < .011
                for row, item in zip(arm['samples'], items):
                    expected_id = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                    assert row['sample_id'] == expected_id
                    text = row['prediction']
                    if task in ('trec', 'triviaqa', 'samsum', 'lsht'):
                        text = text.lstrip('\n').split('\n')[0]
                    score = 100 * max(bench.DATASET_TO_METRIC[task](
                        text, answer, all_classes=item.get('all_classes', []))
                        for answer in item['answers'])
                    assert abs(score - row['score']) < 1e-8
                    assert 0 < row['final_resident_kv_payload_bytes'] <= 512 * 1024 * 1024
                    assert row['prefill_s'] > 0 and row['prefill_peak_alloc_bytes'] > 0
            assert [r['sample_id'] for r in a['samples']] == [r['sample_id'] for r in b['samples']]
            paired[task] = {'tieredkv': b, 'proxy': a}
            differences = [x['score'] - y['score'] for x, y in zip(b['samples'], a['samples'])]
            measures[task] = {
                'sample_count': len(items), 'paired_score_differences': differences,
                'regressed_examples': sum(x < 0 for x in differences),
                'median_prefill_s': {name: statistics.median(r['prefill_s'] for r in arm['samples'])
                                     for name, arm in (('proxy', a), ('actual_query', b))},
                'median_prefill_peak_alloc_bytes': {
                    name: statistics.median(r['prefill_peak_alloc_bytes'] for r in arm['samples'])
                    for name, arm in (('proxy', a), ('actual_query', b))},
                'paid_writes': {name: arm['cache_stats']['paid_writes']
                                for name, arm in (('proxy', a), ('actual_query', b))}}
    tasks = tuple(paired)
    paid_proxy = sum(x['paid_writes']['proxy'] for x in measures.values())
    paid_actual = sum(x['paid_writes']['actual_query'] for x in measures.values())
    total = sum(x['sample_count'] for x in measures.values())
    report = {'input_sha256': refs, 'script_sha256': sha(Path(__file__)),
              'runtime': canonical['runtime'], 'tasks': tasks,
              'quality': compare(paired, 'proxy', tasks=tasks), 'measures': measures,
              'total_regressed_examples': sum(x['regressed_examples'] for x in measures.values()),
              'paid_demotion_writes': {'proxy': paid_proxy, 'actual_query': paid_actual,
                                       'reduction_pct': 100 * (1 - paid_actual / paid_proxy)},
              'scope': f'Selected {total} examples, six tasks, one model/seed; '
                       f'per-task counts { {t: m["sample_count"] for t, m in measures.items()} }. '
                       'Paired bootstrap conditions on these samples and is not a held-out result.',
              'decision': 'Descriptive quality and logical-write comparison only. '
                          'Default promotion requires full-loop latency, stress and held-out gates.'}
    if args.full or args.lru_full:
        manifest = DIR / 'model_sha256.txt'
        report['model_manifest_sha256'] = sha(manifest)
        report['model_manifest_note'] = ('Model files were hashed during the full run; '
                                         'their on-disk modification times predate all arms.')
        if args.full:
            incident = DIR / 'concurrent_gpu_load.json'
            report['concurrent_gpu_load_sha256'] = sha(incident)
            report['full_run_performance_comparison_valid'] = False
            report['performance_note'] = ('An external Ollama process used GPU0 during candidate '
                                          'evaluation. Full-run elapsed and peak differences are '
                                          'confounded; use the isolated ABBA prefill benchmark instead.')
        else:
            report['full_run_performance_comparison_valid'] = False
            report['performance_note'] = ('Sequential task runs are not a paired latency benchmark; '
                                          'use same-process ABBA for timing claims.')
    atomic_json(output, report)
    print(json.dumps({'quality': report['quality'],
                      'total_regressed_examples': report['total_regressed_examples']}, indent=2))


if __name__ == '__main__':
    main()
