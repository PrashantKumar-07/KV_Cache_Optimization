"""Check the opt-in incremental-index pilot and its paired timing gate."""
import hashlib
import json
import statistics
from pathlib import Path

from provenance import atomic_json


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/page_recall'


def read(name):
    path = DIR / name
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    quality, qhash = read('indexed_quality_2x2.json')
    rebuilt, rhash = read('centroid_quality_2x2.json')
    timing, thash = read('indexed_latency_abba_pilot.json')
    assert quality['config']['strategy'] == 'indexed_centroid'
    assert quality['config']['page_index_sha256'] == hashlib.sha256(
        (ROOT / 'experiments/stable_page_index.py').read_bytes()).hexdigest()
    assert timing['config']['order'] == ['query', 'indexed', 'indexed', 'query']
    assert timing['config']['indexed'] and timing['config']['blocks'] == 1
    # Historical timing script hash: a short-lived ABBA version-comparison
    # flag was later removed after its candidate failed. Raw config preserves
    # the executed hash; the current script still supports --indexed.
    assert timing['config']['script_sha256'] == (
        'df57f3136a326b68b7efa0657ef3e72a8c191c31e4c06a2f1970ab84c82b8aa7')
    scores = {}
    for task in ('qasper', 'gov_report'):
        a = quality['results'][task]['tieredkv']
        b = rebuilt['results'][task]['tieredkv']
        assert [r['sample_id'] for r in a['samples']] == [r['sample_id'] for r in b['samples']]
        assert all(0 < r['final_resident_kv_payload_bytes'] <= 512*2**20
                   and r['page_index_logical_bytes'] > 0
                   and r['page_index_changed_key_bytes'] > 0 for r in a['samples'])
        scores[task] = {'indexed_score': a['score'], 'rebuilt_score': b['score'],
                        'indexed_paid_writes': a['cache_stats']['paid_writes'],
                        'rebuilt_paid_writes': b['cache_stats']['paid_writes'],
                        'max_logical_index_mib': max(r['page_index_logical_bytes']
                                                    for r in a['samples']) / 2**20,
                        'changed_key_mib': sum(r['page_index_changed_key_bytes']
                                               for r in a['samples']) / 2**20}
    arms = {name: [value for key, value in timing['results'].items()
                   if key.endswith(name)] for name in ('query', 'indexed')}
    lat = {name: {'median_itl_ms': statistics.median(x['itl_ms'] for x in entries),
                  'median_ttft_ms': statistics.median(x['ttft_ms'] for x in entries),
                  'prefill_peak_gb': max(x['prefill_peak_memory_gb'] for x in entries),
                  'decode_peak_gb': max(x['peak_memory_gb'] for x in entries)}
           for name, entries in arms.items()}
    out = {'input_sha256': {'indexed_quality_2x2.json': qhash,
                            'centroid_quality_2x2.json': rhash,
                            'indexed_latency_abba_pilot.json': thash},
           'scores': scores, 'latency': lat,
           'indexed_itl_change_pct': 100*(lat['indexed']['median_itl_ms'] /
                                           lat['query']['median_itl_ms'] - 1),
           'scope': 'Two examples/task; one 16K prompt, one same-process ABBA block. '
                    'Index parity checked after every synthetic transition and final real requests. '
                    'Logical index bytes exclude Python dictionary overhead; GPU-resident cold K remains. '
                    'Timing script had a temporary version-comparison flag subsequently removed; '
                    'the raw config records its historical source hash.'}
    atomic_json(DIR / 'indexed_report.json', out)
    print(json.dumps({'scores': scores, 'indexed_itl_change_pct':
                      out['indexed_itl_change_pct']}, indent=2))


if __name__ == '__main__':
    main()
