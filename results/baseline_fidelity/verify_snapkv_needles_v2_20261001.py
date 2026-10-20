"""Independent identity/retention/output checks for frozen needle diagnostic."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('results/baseline_fidelity')
raw = root / 'snapkv_needles_v2_20261001.json'
manifest = Path('results/hardware/batched_scores_boundary_retrieval_20261001/manifest.json')
prior = Path('results/hardware/batched_scores_boundary_retrieval_20261001/verified_report.json')
output = root / 'snapkv_needles_verified_v2_20261001.json'
bundle = root / 'snapkv_needles_source_v2_20261001.tar.gz'
assert not output.exists() and not bundle.exists(), 'Preserve prior evidence'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
r = json.loads(raw.read_text()); m = json.loads(manifest.read_text()); p = json.loads(prior.read_text())
assert r['passed'] and r['resource_matched'] is False and r['manifest_sha256'] == sha(manifest)
for path, expected in r['source_sha256'].items():
    assert sha(path) == expected, path
assert sha(root / 'check_snapkv_needles_executed_v2_20261001.py') == r['source_sha256']['experiments/check_snapkv_needles.py']
cases = {c['name']: c for c in m['cases'] if c.get('oracle')}
previous = {c['case']: c for c in p['retrieval_results']}
assert len(r['rows']) == len(cases) == len(previous) == 6
assert {row['case'] for row in r['rows']} == set(cases)
rows = []
for row in r['rows']:
    case = cases[row['case']]
    assert row['token_sha256'] == case['token_sha256']
    assert row['prompt_tokens'] == len(case['ids'])
    assert row['answer_correct'] == (case['answer'].casefold() in row['prediction'].casefold())
    assert 0 < len(row['generated_token_ids']) <= case['max_gen']
    assert row['final_owned_kv_payload_bytes'] == (1024 + len(row['generated_token_ids']) - 1) * 32 * 32 * 128 * 2 * 2
    layers = row['retained_answer_by_layer']
    assert len(layers) == 32 and [x['layer'] for x in layers] == list(range(32))
    for layer in layers:
        counts = layer['retained_tokens_per_head']
        span = case['answer_span'][1] - case['answer_span'][0]
        assert layer['query_heads'] == len(counts) == 32 and layer['answer_span_tokens'] == span
        assert all(0 <= n <= span for n in counts)
        assert layer['complete_span_heads'] == sum(n == span for n in counts)
    assert row['complete_span_query_heads'] == sum(x['complete_span_heads'] for x in layers)
    prev = previous[row['case']]
    assert prev['gpu_correct'] is False and prev['cpu_correct'] is False and prev['oracle_correct'] is True
    rows.append(dict(case=row['case'], snapkv_correct=row['answer_correct'],
        tiered_gpu_correct=prev['gpu_correct'], tiered_cpu_correct=prev['cpu_correct'],
        full_cache_correct=prev['oracle_correct'], complete_span_layer_head_pairs=row['complete_span_query_heads'],
        layer_head_pairs=1024, owned_kv_payload_bytes=row['final_owned_kv_payload_bytes']))
assert r['correct_cases'] == sum(x['snapkv_correct'] for x in rows)
with tarfile.open(bundle, 'w:gz') as archive:
    for path in list(r['source_sha256']) + ['results/baseline_fidelity/verify_snapkv_needles_v2_20261001.py']:
        archive.add(path, arcname=path)
report = dict(identity_and_internal_consistency_passed=True, raw_sha256=sha(raw),
    manifest_sha256=sha(manifest), prior_report_sha256=sha(prior), source_bundle_sha256=sha(bundle),
    correct_cases=r['correct_cases'], rows=rows, resource_matched=False,
    limitations='Reused six synthetic development cases; no independent quality or speed/memory gate. SnapKV expanded 32-head growing cache differs from Tiered 8-head fixed tiers. Correlation supports investigating retention, not isolated causal proof or selector promotion.')
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
