"""Recompute coordinator gates from immutable executed sources and raw rows."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def bundle(version):
    manifest = read(f'{version}_manifest.json')
    path = ROOT / f'{version}.tar.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['bundle_sha256']
    with tarfile.open(path) as archive:
        sources = {name: archive.extractfile(name).read()
                   for name in manifest['source_sha256']}
    for name, value in sources.items():
        assert hashlib.sha256(value).hexdigest() == manifest['source_sha256'][name]
    return manifest['source_sha256'], sources


def exact(reference, candidate, frames):
    for field in ('state_sha256', 'history_sha256', 'every_visible_state_sha256',
                  'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes',
                  'initial_d2h_payload_bytes', 'occupied_host_bytes', 'reserved_host_bytes',
                  'native_d2h_calls', 'native_h2d_calls', 'per_layer_initial_d2h_calls'):
        assert reference[field] == candidate[field], field
    assert len(reference['every_visible_state_sha256']) == frames
    assert len(reference['trials']) == len(candidate['trials']) == 1
    for field in ('predicted_token_ids', 'paid_writes', 'writes_saved', 'demoted',
                  'promoted', 'decode_steps'):
        assert reference['trials'][0][field] == candidate['trials'][0][field], field


def provenance(report, hashes):
    assert report['runner_sha256'] == hashes['experiments/compare_cpu_batched.py']
    for name, expected in report['eviction_backend_sha256'].items():
        assert hashes[f'experiments/{name}'] == expected
    for key, name in (('backend_sha256', 'cpu_batched_cache'),
                      ('slot_backend_sha256', 'cpu_slot_cache'),
                      ('query_capture_sha256', 'evaluate_query_selection'),
                      ('score_backend_sha256', 'batched_score_policy')):
        assert report[key] == hashes[f'experiments/{name}.py']


def main():
    hashes2, sources2 = bundle('source_v2')
    hashes3, sources3 = bundle('source_v3')
    for name in ('src/tiered_kv_cache.py', 'experiments/batched_eviction_policy.py',
                 'experiments/direct_shared_pool.py', 'experiments/probe_batched_eviction_selection.py',
                 'experiments/probe_protected_mask_numpy.py'):
        assert hashes2[name] == hashes3[name], name
    methods = lambda source: {n.name: n for n in ast.walk(ast.parse(source))
                              if isinstance(n, ast.FunctionDef)}
    old = methods((ROOT / 'tiered_kv_cache_before.py').read_bytes())
    now = methods(sources3['src/tiered_kv_cache.py'])
    expanded = copy.deepcopy(now['evict_and_demote'])
    matches = [i for i, n in enumerate(expanded.body) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == '_select_hot_victims']
    assert len(matches) == 1
    i = matches[0]
    expanded.body[i:i+1] = copy.deepcopy(now['_select_hot_victims'].body[2:-1])
    assert ast.dump(expanded) == ast.dump(old['evict_and_demote'])
    synthetic = read('synthetic_v2.json')
    assert synthetic['passed'] and len(synthetic['checks']) == 32
    for name, expected in synthetic['source_sha256'].items():
        assert hashes2[name] == expected
    assert all(r['exact_state_payload_counters_and_cleanup'] and r['appended_tokens'] == 18
               and r['last_query_attention_summary'] for r in synthetic['checks'])
    parity = read('parity.json')
    assert not parity['passed'] and len(parity['rows']) == 6
    assert '30907' in parity['error'] and '31500' in parity['error']
    provenance(parity, hashes2)
    for i, ctx in enumerate((4096, 8192, 16384)):
        a, b = parity['rows'][2*i:2*i+2]
        assert a['context'] == b['context'] == ctx
        exact(a, b, 6)
    maximum = read('max_parity.json')
    provenance(maximum, hashes3)
    assert maximum['passed'] and len(maximum['rows']) == 2
    assert all(r['context'] == 31500 for r in maximum['rows'])
    assert maximum['prompt_sha256'] == hashlib.sha256((ROOT/'max_context_prompt.txt').read_bytes()).hexdigest()
    exact(*maximum['rows'], 6)
    abba = read('abba.json')
    provenance(abba, hashes3)
    assert abba['passed'] and len(abba['rows']) == 12
    assert abba['config']['warmup'] == 1 and abba['config']['decode'] == 50
    rows = abba['rows']
    control, candidate = 'direct_scores', 'eviction_scores'
    for block in range(3):
        group = rows[4*block:4*block+4]
        assert [r['arm'] for r in group] == [control, candidate, candidate, control]
        assert all(r['block'] == block and r['context'] == 16384 for r in group)
    for row in rows[1:]:
        exact(rows[0], row, 0)
    summary = {}
    for arm in (control, candidate):
        chosen = [r for r in rows if r['arm'] == arm]
        steps = [v for r in chosen for v in r['trials'][0]['step_ms']]
        assert len(steps) == 300
        summary[arm] = dict(median_ms=statistics.median(steps),
                            p95_ms=float(np.percentile(steps, 95)),
                            ttft_ms=statistics.median(r['ttft_ms'] for r in chosen),
                            prefill_peak_gb=max(r['prefill_peak_memory_gb'] for r in chosen))
    a, b = summary[control], summary[candidate]
    blocks = [{arm: statistics.median(v for r in rows if r['block'] == block and r['arm'] == arm
                                    for v in r['trials'][0]['step_ms'])
               for arm in (control, candidate)} for block in range(3)]
    gates = dict(median_gain_5pct=b['median_ms'] <= .95*a['median_ms'],
                 p95_regression_at_most_2pct=b['p95_ms'] <= 1.02*a['p95_ms'],
                 all_blocks_faster=all(r[candidate] < r[control] for r in blocks),
                 ttft_regression_at_most_10pct=b['ttft_ms'] <= 1.1*a['ttft_ms'],
                 prefill_peak_regression_at_most_2pct=b['prefill_peak_gb'] <= 1.02*a['prefill_peak_gb'])
    output = dict(correctness_passed=True, performance_gates=gates,
                  all_performance_gates_passed=all(gates.values()), summary=summary, blocks=blocks,
                  decode_gain_pct=100*(1-b['median_ms']/a['median_ms']),
                  ttft_change_pct=100*(b['ttft_ms']/a['ttft_ms']-1),
                  synthetic_cases=32, context_boundaries=[4096,8192,16384,31500],
                  extraction_whole_body_ast_equal=True,
                  limitations='Fixed teacher stream, one model and seed; free generation and interleaved coordinator still pending. No SnapKV, physical STT, quality superiority, serving throughput or 10% model-memory claim.',
                  evidence_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                   for name in ('parity.json','max_parity.json','synthetic.json',
                                                'synthetic_v2.json','abba.json','source_v2.tar.gz','source_v3.tar.gz')})
    destination = ROOT/'verified_report.json'
    assert not destination.exists(), 'Preserve prior verification'
    destination.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
