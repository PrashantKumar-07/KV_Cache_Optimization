"""Summarize same-process ABBA full-loop latency for actual-query placement."""
import hashlib
import json
import statistics
import argparse
from pathlib import Path

import numpy as np
from provenance import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/baseline_fidelity'
QUERY_SCRIPT_SHA256 = 'c1d6a45ad7002eb13efddf23fc15304613835978c3c776a69ba1190560e26c12'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lru', action='store_true')
    args = parser.parse_args()
    raw = DIR / ('real_q_lru_full_loop_abba.json' if args.lru else 'real_q_full_loop_abba.json')
    out = DIR / ('real_q_lru_full_loop_abba_report.json' if args.lru
                 else 'real_q_full_loop_abba_report.json')
    doc = json.loads(raw.read_text())
    cfg, results = doc['config'], doc['results']
    assert cfg['compare_prefill'] and cfg['blocks'] == 3
    assert cfg['order'] == ['proxy', 'actual', 'actual', 'proxy'] * 3
    assert cfg.get('prefill_strategy', 'query') == ('lru' if args.lru else 'query')
    expected_script_hash = (hashlib.sha256((ROOT / 'experiments/compare_query_latency.py')
                            .read_bytes()).hexdigest() if args.lru else QUERY_SCRIPT_SHA256)
    assert cfg['comparison_script_sha256'] == expected_script_hash
    report = {'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'config': cfg, 'blocks': [],
              'scope': 'One 16K real-text prompt, fixed 50-token decode, three same-process '
                       'ABBA blocks. Timing is completed full model+policy loop, not throughput.'}
    for block in range(3):
        rows = {arm: [results[f'{i}_{arm}'] for i in range(block*4, block*4+4)
                      if cfg['order'][i] == arm] for arm in ('proxy', 'actual')}
        assert all(len(v) == 2 for v in rows.values())
        summary = {}
        for arm, entries in rows.items():
            steps = [step for entry in entries for trial in entry['trials']
                     for step in trial['step_ms']]
            summary[arm] = {'median_itl_ms': float(np.median(steps)),
                            'p95_itl_ms': float(np.quantile(steps, .95)),
                            'median_ttft_ms': statistics.median(e['ttft_ms'] for e in entries),
                            'max_prefill_peak_gb': max(e['prefill_peak_memory_gb'] for e in entries),
                            'max_decode_peak_gb': max(e['peak_memory_gb'] for e in entries)}
        summary['median_itl_change_pct'] = 100 * (
            summary['actual']['median_itl_ms']/summary['proxy']['median_itl_ms'] - 1)
        summary['p95_itl_change_pct'] = 100 * (
            summary['actual']['p95_itl_ms']/summary['proxy']['p95_itl_ms'] - 1)
        report['blocks'].append(summary)
    report['median_block_itl_change_pct'] = statistics.median(
        b['median_itl_change_pct'] for b in report['blocks'])
    report['median_block_p95_change_pct'] = statistics.median(
        b['p95_itl_change_pct'] for b in report['blocks'])
    atomic_json(out, report)
    print(json.dumps({'median_block_itl_change_pct': report['median_block_itl_change_pct'],
                      'median_block_p95_change_pct': report['median_block_p95_change_pct']}, indent=2))


if __name__ == '__main__':
    main()
