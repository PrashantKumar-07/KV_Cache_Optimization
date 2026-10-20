"""Independently audit the frozen six-probe head-reduction screen."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = json.loads((BASE / 'result.json').read_text())
    manifest = json.loads((BASE / 'manifest.json').read_text())
    frozen = json.loads((BASE / 'source_manifest.json').read_text())['source_sha256']
    prior_path = ROOT / 'results/baseline_fidelity/tiered_query_needles_v2_20261001.json'
    prior = json.loads(prior_path.read_text())
    assert raw['passed'] and raw['manifest_sha256'] == sha(BASE / 'manifest.json')
    assert raw['prior_sha256'] == sha(prior_path)
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        for name, digest in raw['source_sha256'].items():
            assert digest == frozen[name]
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
    old = {r['case']: r for r in prior['rows'] if r['arm'] == 'actual_query'}
    cases = {c['name']: c for c in manifest['cases'] if c.get('oracle')}
    assert len(raw['rows']) == 12 and len(cases) == 6
    counts = dict(sum_heads=0, max_head=0)
    retained = dict(sum_heads=0, max_head=0)
    details = []
    for name, case in cases.items():
        assert hashlib.sha256(json.dumps(case['ids'], separators=(',', ':')).encode()).hexdigest() == case['token_sha256']
        pair = {r['arm']: r for r in raw['rows'] if r['case'] == name}
        assert set(pair) == set(counts)
        assert pair['sum_heads']['initial_cache_rows'] == pair['max_head']['initial_cache_rows']
        for arm, row in pair.items():
            assert row['token_sha256'] == case['token_sha256']
            assert row['answer_correct'] == (case['answer'].casefold() in row['prediction'].casefold())
            assert 0 < len(row['generated_token_ids']) <= case['max_gen']
            assert len(row['initial_answer_retention']) == len(row['initial_cache_rows']) == 32
            complete = sum(x['retained_tokens'] == x['span_tokens'] for x in row['initial_answer_retention'].values())
            assert row['complete_span_layers'] == complete
            for layer in row['initial_cache_rows'].values():
                assert layer['hot_rows'] <= 1024 and layer['cold_rows'] <= 2048
                assert layer['hot_payload_bytes'] == layer['hot_rows'] * 8 * 128 * 2 * 2
                assert layer['cold_payload_bytes'] == layer['cold_rows'] * 8 * 128 * 2 * 2
            if arm == 'sum_heads':
                for field in ('generated_token_ids', 'prediction', 'state_sha256', 'trajectory_sha256',
                              'trajectory_frames', 'initial_answer_retention', 'counters', 'prefill_query_shapes'):
                    assert row[field] == old[name][field], (name, field)
            counts[arm] += row['answer_correct']
            retained[arm] += complete
        details.append(dict(case=name, correct={a:r['answer_correct'] for a,r in pair.items()},
                            complete_span_layers={a:r['complete_span_layers'] for a,r in pair.items()}))
    assert counts == raw['correct_cases'] and retained == raw['complete_span_layers']
    advance = counts['max_head'] >= 4 and retained['max_head'] > retained['sum_heads']
    assert raw['advance_to_broader_development'] == advance
    report = dict(verified=True, raw_sha256=sha(BASE / 'result.json'),
                  verifier_sha256=sha(Path(__file__)), source_archive_sha256=sha(BASE / 'source.tar.gz'),
                  correct_cases=counts, complete_span_layers=retained, cases=details,
                  advance_to_broader_development=advance,
                  limitation='Reused development probes. Both methods already answer all six correctly; two additional complete layer spans do not establish quality superiority.')
    output = BASE / 'verified_report.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
