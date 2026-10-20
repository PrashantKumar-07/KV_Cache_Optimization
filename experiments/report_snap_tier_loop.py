"""Validate and summarize the preregistered SnapKV/TieredKV loop comparison."""
import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    config, results = raw['config'], raw['results']
    assert config['blocks'] == 3
    assert config['order'] == ['snap', 'tiered', 'tiered', 'snap'] * 3
    assert len(results) == 12
    assert config['ctx_len'] == 16384 and config['decode'] == 50
    assert config['configured_max_final_kv_mib'] == 512
    arms = {'snap': [], 'tiered': []}
    for i, name in enumerate(config['order']):
        result = results[f'{i}_{name}']
        assert len(result['trials']) == 2
        assert all(len(t['step_ms']) == 50 for t in result['trials'])
        assert result['final_kv_payload_bytes'] <= 512 * 2**20
        arms[name].append(result)
    summary = {}
    for name, runs in arms.items():
        steps = [s for run in runs for trial in run['trials'] for s in trial['step_ms']]
        summary[name] = {
            'median_itl_ms': median(steps),
            'p95_itl_ms': float(np.percentile(steps, 95)),
            'median_ttft_ms': median(t['ttft_ms'] for run in runs for t in run['trials']),
            'max_prefill_peak_gb': max(run['prefill_peak_memory_gb'] for run in runs),
            'max_decode_peak_gb': max(run['peak_memory_gb'] for run in runs),
            'final_kv_payload_mib': median(run['final_kv_payload_bytes'] / 2**20 for run in runs),
            'block_median_itl_ms': [median(s for run in runs[2*b:2*b+2]
                for trial in run['trials'] for s in trial['step_ms']) for b in range(3)],
        }
        if name == 'tiered':
            summary[name]['median_forward_ms'] = median(s for run in runs
                for trial in run['trials'] for s in trial['forward_ms'])
    report = {'config': config, 'summary': summary,
              'tiered_over_snap_itl_ratio': summary['tiered']['median_itl_ms'] /
                                            summary['snap']['median_itl_ms'],
              'limitations': ['One fixed prompt and 50 forced decode forwards',
                              'Configured maximum final KV payload matched; actual bytes differ',
                              'Tiered requires eager attention weights; Snap does not',
                              'PyTorch allocation peaks are not physical STT-RAM measurements']}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'tiered_over_snap_itl_ratio':
                      report['tiered_over_snap_itl_ratio']}, indent=2))


if __name__ == '__main__':
    main()
