"""Summarize the same-process ABBA actual-query prefill benchmark."""
import hashlib
import json
import statistics
from pathlib import Path

from provenance import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/baseline_fidelity/real_q_prefill_abba.json'
OUT = ROOT / 'results/baseline_fidelity/real_q_prefill_abba_report.json'


def main():
    raw = json.loads(RAW.read_text())
    report = {'raw_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime': raw['runtime'], 'protocol': raw['protocol'], 'tasks': {},
              'scope': 'Single prompt per task; three same-process ABBA blocks after warmup. '
                       'Measured prefill plus placement only, not full-loop decode or TTFT.'}
    for task, data in raw['tasks'].items():
        rows = data['rows']
        assert len(rows) == 12
        assert all([r['arm'] for r in rows[i:i+4]] ==
                   ['proxy', 'actual_query', 'actual_query', 'proxy']
                   for i in (0, 4, 8))
        output = {'prompt_tokens': data['prompt_tokens'], 'blocks': {}}
        for metric in ('seconds', 'peak_alloc_bytes'):
            blocks = []
            for block in range(3):
                sample = [r for r in rows if r['block'] == block]
                assert len(sample) == 4
                baseline = statistics.median(r[metric] for r in sample if r['arm'] == 'proxy')
                candidate = statistics.median(r[metric] for r in sample if r['arm'] == 'actual_query')
                blocks.append({'proxy': baseline, 'actual_query': candidate,
                               'relative_change_pct': 100 * (candidate / baseline - 1)})
            output['blocks'][metric] = blocks
            output[metric] = {
                'proxy_median_of_block_medians': statistics.median(x['proxy'] for x in blocks),
                'actual_query_median_of_block_medians':
                    statistics.median(x['actual_query'] for x in blocks),
                'median_relative_change_pct': statistics.median(
                    x['relative_change_pct'] for x in blocks)}
        report['tasks'][task] = output
    atomic_json(OUT, report)
    print(json.dumps(report['tasks'], indent=2))


if __name__ == '__main__':
    main()
