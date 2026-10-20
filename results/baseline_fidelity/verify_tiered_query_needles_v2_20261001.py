"""Check frozen controls, actual-query consumption, retention and equal initial KV."""
import hashlib
import json
from pathlib import Path
import tarfile
root = Path('results/baseline_fidelity')
raw = root / 'tiered_query_needles_v2_20261001.json'
manifest = Path('results/hardware/batched_scores_boundary_retrieval_20261001/manifest.json')
prior_path = manifest.with_name('result.json')
output = root / 'tiered_query_needles_verified_v2_20261001.json'
bundle = root / 'tiered_query_needles_source_v2_20261001.tar.gz'
assert not output.exists() and not bundle.exists(), 'Preserve prior evidence'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
r = json.loads(raw.read_text()); m = json.loads(manifest.read_text()); prior = json.loads(prior_path.read_text())
assert r['passed'] and r['initial_cache_geometry_and_payload_equal'] and r['control_reproduces_frozen_boundary_run']
assert r['manifest_sha256'] == sha(manifest) and r['prior_raw_sha256'] == sha(prior_path)
for path, expected in r['source_sha256'].items():
    assert sha(path) == expected, path
assert sha(root / 'check_tiered_query_needles_executed_v2_20261001.py') == r['source_sha256']['experiments/check_tiered_query_needles.py']
runtime_files = sorted(Path('src').glob('*.py')) + [Path('experiments')/p for p in ('longbench_eval.py','longbench_metrics.py','latency_profile.py','ppl_eval.py','provenance.py')]
h = hashlib.sha256()
for path in runtime_files:
    h.update(str(path).encode()); h.update(path.read_bytes())
assert h.hexdigest() == r['runtime']['source_sha256']
cases = {c['name']:c for c in m['cases'] if c.get('oracle')}
controls = {row['case']:row for row in prior['rows'] if row['arm'] == 'gpu'}
assert len(r['rows']) == 12 and len(cases) == 6
rows = []
for name, case in cases.items():
    pair = {row['arm']: row for row in r['rows'] if row['case'] == name}
    assert set(pair) == {'proxy','actual_query'}
    a,b = pair['proxy'],pair['actual_query']
    for row in pair.values():
        assert row['token_sha256'] == case['token_sha256']
        assert row['answer_correct'] == (case['answer'].casefold() in row['prediction'].casefold())
        assert len(row['initial_answer_retention']) == len(row['initial_cache_rows']) == 32
        assert 0 < len(row['generated_token_ids']) <= case['max_gen']
        for state in row['initial_cache_rows'].values():
            assert state == dict(hot_rows=1024,cold_rows=1536,hot_payload_bytes=4194304,cold_payload_bytes=6291456,visible_rows=1056)
        for state in row['initial_answer_retention'].values():
            assert state['span_tokens'] == case['answer_span'][1]-case['answer_span'][0] == 12
            assert 0 <= state['retained_tokens'] <= 12
            assert state['hot_tokens']+state['cold_tokens'] == state['retained_tokens']
    assert a['initial_cache_rows'] == b['initial_cache_rows']
    for field in ('generated_token_ids','prediction','state_sha256','trajectory_sha256','trajectory_frames','initial_answer_retention','counters'):
        assert a[field] == controls[name][field], field
    assert a['prefill_query_shapes'] == {}
    assert set(b['prefill_query_shapes']) == {str(i) for i in range(32)}
    assert all(shape == [32,128,128] for shape in b['prefill_query_shapes'].values())
    rows.append(dict(case=name,prompt_tokens=len(case['ids']),proxy_correct=a['answer_correct'],actual_query_correct=b['answer_correct'],
        proxy_complete_span_layers=sum(x['retained_tokens']==12 for x in a['initial_answer_retention'].values()),
        actual_query_complete_span_layers=sum(x['retained_tokens']==12 for x in b['initial_answer_retention'].values()),
        initial_hot_plus_cold_kv_payload_bytes=sum(x['hot_payload_bytes']+x['cold_payload_bytes'] for x in b['initial_cache_rows'].values())))
assert r['correct_cases'] == {arm:sum(row['answer_correct'] for row in r['rows'] if row['arm']==arm) for arm in ('proxy','actual_query')}
files = set(map(str,runtime_files)) | set(r['source_sha256']) | {'experiments/batched_score_policy.py','experiments/cpu_batched_cache.py','experiments/cpu_slot_cache.py','results/baseline_fidelity/verify_tiered_query_needles_v2_20261001.py'}
with tarfile.open(bundle,'w:gz') as archive:
    for path in sorted(files):
        archive.add(path,arcname=path)
report=dict(identity_control_consumption_retention_and_initial_payload_verified=True,raw_sha256=sha(raw),source_bundle_sha256=sha(bundle),manifest_sha256=sha(manifest),
    cases=6,correct_cases=r['correct_cases'],rows=rows,initial_kv_payload_equal=True,
    limitations='Reused development probes; actual-query capture/scoring has extra prefill work/memory. Final free-generation resources differ. No independent quality, matched total-resource or latency gate. Prior 24-document holdout quality interval crossed zero; earlier two-prompt prefill guard passed. Raw runner wording about broader factual/prefill rejection is imprecise: option remains experimental because broader replication/joint gates are unproven, not because every prefill guard failed.')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
