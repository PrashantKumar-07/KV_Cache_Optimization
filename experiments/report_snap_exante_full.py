"""Verify frozen 135-example SnapKV comparison at equal configured KV ceiling."""
import hashlib
import json
import argparse
from pathlib import Path

import longbench_eval as bench
from provenance import atomic_json
from report_remediation import compare


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/baseline_fidelity'
SOURCES = {
    'archived': (
        ('snap_exante_full_qn25.json', 'real_q_lru_rescore_125.json', ('qasper', 'narrativeqa')),
        ('snap_exante_full_multi25.json', 'real_q_lru_rescore_125.json', ('multifieldqa_en',)),
        ('snap_exante_full_ht25.json', 'real_q_lru_rescore_125.json', ('hotpotqa', 'triviaqa')),
        ('snap_exante_full_gov10.json', 'real_q_lru_rescore_gov10.json', ('gov_report',)),
    ),
    'currentbool': (
        ('currentbool_snap_qn25.json', 'currentbool_tier125.json', ('qasper', 'narrativeqa')),
        ('currentbool_snap_multi25.json', 'currentbool_tier125.json', ('multifieldqa_en',)),
        ('currentbool_snap_ht25.json', 'currentbool_tier125.json', ('hotpotqa', 'triviaqa')),
        ('currentbool_snap_gov10.json', 'currentbool_tiergov10.json', ('gov_report',)),
    ),
    'currenthot': (
        ('currenthot_snap_qn25.json', 'currenthot_tier125.json', ('qasper', 'narrativeqa')),
        ('currenthot_snap_multi25.json', 'currenthot_tier125.json', ('multifieldqa_en',)),
        ('currenthot_snap_ht25.json', 'currenthot_tier125.json', ('hotpotqa', 'triviaqa')),
        ('currenthot_snap_gov10.json', 'currenthot_tiergov10.json', ('gov_report',)),
    ),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-set', choices=SOURCES, default='archived')
    args = parser.parse_args()
    paired, resources, refs = {}, {}, {}
    model = runtime = None
    for snap_name, tier_name, tasks in SOURCES[args.source_set]:
        sp, tp = DIR / snap_name, DIR / tier_name
        snap, tier = json.loads(sp.read_text()), json.loads(tp.read_text())
        refs[sp.name], refs[tp.name] = sha(sp), sha(tp)
        c, tc = snap['config'], tier['config']
        assert c['attn_implementation'] == 'sdpa' and c['window'] == 32
        assert tc['strategy'] == 'lru' and tc['real_query_prefill'] is True
        assert tc['tiered_vram_budget'] == 992 and tc['tiered_stt_budget'] == 3104
        assert tc['stt_expose_quota'] == 32
        if model is None:
            model, runtime = c['model'], c['runtime']
        assert c['model'] == tc['model'] == model
        assert c['runtime'] == tc['runtime'] == runtime
        if args.source_set == 'archived':
            old_tier = json.loads((DIR / ('real_q_lru_candidate_gov10.json'
                if 'gov_report' in tasks else 'real_q_lru_candidate_125.json')).read_text())
        assert c['seed'] == tc['seed'] == 0
        assert c['max_samples'] == (10 if 'gov_report' in tasks else 25)
        assert c['capacity'] == 1025 - bench.DATASET_TO_MAXGEN[tasks[0]]
        for task in tasks:
            assert c['capacity'] == 1025 - bench.DATASET_TO_MAXGEN[task]
            assert c['dataset_sha256'][task] == tc['dataset_sha256'][task]
            assert sha(bench.task_data_path(c['data_dir'], task)) == c['dataset_sha256'][task]
            assert task not in paired
            a = tier['results'][task]['tieredkv']
            b = snap['results'][task]['snapkv_upstream_selector']
            if args.source_set == 'archived':
                old_a = old_tier['results'][task]['tieredkv']
                assert a['cache_stats'] == old_a['cache_stats']
                assert [(r['sample_id'], r['prediction'], r['score'],
                         r['final_resident_kv_payload_bytes']) for r in a['samples']] == [
                         (r['sample_id'], r['prediction'], r['score'],
                          r['final_resident_kv_payload_bytes']) for r in old_a['samples']]
            assert a['n'] == b['n'] == c['max_samples']
            assert [r['sample_id'] for r in a['samples']] == [r['sample_id'] for r in b['samples']]
            ratios = []
            for ar, br in zip(a['samples'], b['samples']):
                assert 0 < ar['final_resident_kv_payload_bytes'] <= 512*2**20
                expected = (min(br['prompt_tokens'], c['capacity']) +
                            br['generated_tokens'] - 1) * 524288
                assert 0 < br['final_kv_payload_bytes'] == expected <= 512*2**20
                ratios.append(expected / ar['final_resident_kv_payload_bytes'])
            paired[task] = {'tieredkv': a, 'snapkv': b}
            sum_t = sum(r['final_resident_kv_payload_bytes'] for r in a['samples'])
            sum_s = sum(r['final_kv_payload_bytes'] for r in b['samples'])
            differences = [x['score'] - y['score'] for x, y in zip(a['samples'], b['samples'])]
            resources[task] = {'n': a['n'], 'snap_capacity': c['capacity'],
                'snap_maxgen': bench.DATASET_TO_MAXGEN[task],
                'tiered_mean_mib': sum_t / a['n'] / 2**20,
                'snap_mean_mib': sum_s / b['n'] / 2**20,
                'ratio_of_mean_final_bytes': sum_s / sum_t,
                'min_per_request_byte_ratio': min(ratios),
                'max_per_request_byte_ratio': max(ratios),
                'tiered_wins': sum(d > 0 for d in differences),
                'tiered_regressions': sum(d < 0 for d in differences),
                'snap_final_attended_rows_range': [
                    min((min(r['prompt_tokens'], c['capacity']) +
                         r['generated_tokens'] - 1) for r in b['samples']),
                    max((min(r['prompt_tokens'], c['capacity']) +
                         r['generated_tokens'] - 1) for r in b['samples'])]}
    tasks = tuple(paired)
    report = {'input_sha256': refs, 'quality': compare(paired, 'snapkv', tasks=tasks),
              'resources': resources, 'n_total': sum(x['n'] for x in resources.values()),
              'source_set': args.source_set,
              'scope': 'Selected 135 examples, one model/seed. Both policies have an ex-ante '
                       '512 MiB final resident KV ceiling. Actual bytes, attended width, '
                       'prefill peak, decode implementation, and physical tiers differ. '
                       'Conditional paired bootstrap is not held-out or SOTA evidence.'}
    assert report['n_total'] == 135
    output_name = {'archived': 'snap_exante_full_report.json',
                   'currentbool': 'snap_exante_full_currentbool_report.json',
                   'currenthot': 'snap_exante_full_currenthot_report.json'}[args.source_set]
    output = DIR / output_name
    atomic_json(output, report)
    print(json.dumps({'quality': report['quality'], 'resources': report['resources']}, indent=2))


if __name__ == '__main__':
    main()
