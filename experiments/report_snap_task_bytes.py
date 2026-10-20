"""Verify a task-mean-byte SnapKV pilot without claiming exact resource parity."""
import argparse
import hashlib
import json
from pathlib import Path

import longbench_eval as bench
from provenance import atomic_json
from report_remediation import compare


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/baseline_fidelity'
FILES = {'qasper': 'snap832_qn10.json',
         'narrativeqa': 'snap832_qn10.json',
         'gov_report': 'snap500_gov5.json'}
EXANTE_FILES = {'qasper': 'snap_exante_897_qn10.json',
                'narrativeqa': 'snap_exante_897_qn10.json',
                'gov_report': 'snap_exante_513_gov5.json'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exante', action='store_true')
    args = parser.parse_args()
    files = EXANTE_FILES if args.exante else FILES
    tier_path = DIR / 'obs128_lru_15.json'
    tier = json.loads(tier_path.read_text())
    refs = {tier_path.name: sha(tier_path)}
    paired, resources = {}, {}
    for task, filename in files.items():
        path = DIR / filename
        snap = json.loads(path.read_text())
        refs[filename] = sha(path)
        c = snap['config']
        assert c['attn_implementation'] == 'sdpa' and c['window'] == 32
        expected_capacity = (1025 - bench.DATASET_TO_MAXGEN[task] if args.exante
                             else 500 if task == 'gov_report' else 832)
        assert c['capacity'] == expected_capacity
        assert c['model'] == tier['config']['model']
        assert c['seed'] == tier['config']['seed'] == 0
        assert c['dataset_sha256'][task] == tier['config']['dataset_sha256'][task]
        assert sha(bench.task_data_path(c['data_dir'], task)) == c['dataset_sha256'][task]
        a = tier['results'][task]['tieredkv']
        b = snap['results'][task]['snapkv_upstream_selector']
        assert a['n'] == b['n'] == 5
        assert [r['sample_id'] for r in a['samples']] == [r['sample_id'] for r in b['samples']]
        assert all(0 < row['final_resident_kv_payload_bytes'] <= 512*2**20
                   for row in a['samples'])
        assert all(row['final_kv_payload_bytes'] ==
                   (min(row['prompt_tokens'], c['capacity']) + row['generated_tokens'] - 1)
                   * 524288 for row in b['samples'])
        if args.exante:
            assert all(row['final_kv_payload_bytes'] <= 512*2**20
                       for row in b['samples'])
        paired[task] = {'tieredkv': a, 'snapkv': b}
        ratios = [x['final_kv_payload_bytes'] / y['final_resident_kv_payload_bytes']
                  for x, y in zip(b['samples'], a['samples'])]
        tier_bytes = sum(r['final_resident_kv_payload_bytes'] for r in a['samples'])
        snap_bytes = sum(r['final_kv_payload_bytes'] for r in b['samples'])
        resources[task] = {'snap_capacity': c['capacity'],
                           'tiered_mean_final_mib': tier_bytes / 5 / 2**20,
                           'snap_mean_final_mib': snap_bytes / 5 / 2**20,
                           'ratio_of_mean_final_bytes': snap_bytes / tier_bytes,
                           'per_request_snap_to_tiered_ratios': ratios,
                           'snap_attended_rows_final': [
                               min(r['prompt_tokens'], c['capacity']) +
                               r['generated_tokens'] - 1 for r in b['samples']],
                           'tiered_attended_row_ceiling': 1024}
    tasks = tuple(files)
    report = {'input_sha256': refs, 'quality': compare(paired, 'snapkv', tasks=tasks),
              'resources': resources,
              'scope': ('Selected first five per task. Both configured final KV ceilings '
                        'are 512 MiB using task max generation known before decoding; '
                        'actual bytes, attended width and prefill peak still differ. '
                        if args.exante else
                        'Selected first five per task, task-level capacities chosen using an '
                        'earlier observed length regime. Mean final KV payload only; '
                        'not ex-ante per-request equal bytes, equal attention width, '
                        'or matched prefill peak. ') +
                       'Tiered uses SDPA prefill/eager decode; Snap adapter uses SDPA '
                       'throughout. No latency/SOTA claim.'}
    atomic_json(DIR / ('snap_exante_512_report.json' if args.exante else
                       'snap_task_bytes_report.json'), report)
    print(json.dumps({'quality': report['quality'],
                      'resources': report['resources']}, indent=2))


if __name__ == '__main__':
    main()
