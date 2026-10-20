"""Validate observed ownership against geometry and frozen source identities."""
import hashlib
import json
from pathlib import Path
import tarfile
root = Path('results/baseline_fidelity')
raw = root / 'snap_tier_resource_trajectory_20261001.json'
output = root / 'snap_tier_resource_trajectory_verified_20261001.json'
bundle = root / 'snap_tier_resource_trajectory_source_20261001.tar.gz'
assert not output.exists() and not bundle.exists(), 'Preserve prior evidence'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
r = json.loads(raw.read_text())
assert r['passed'] and r['gpu_cpu_geometry_and_predictions_equal']
assert r['resource_matched'] is False and len(r['input_token_ids']) == 16384
assert sha('results/shadow_ablation_full/fixed_stream_20260929.json') == r['teacher_source_sha256']
assert hashlib.sha256(Path('results/remediation_v3/latency_prompt.txt').read_text().encode()).hexdigest() == r['prompt_sha256']
for path, expected in r['source_sha256'].items():
    assert sha(path) == expected, path
assert sha(root / 'audit_snap_tier_resources_executed_20261001.py') == r['source_sha256']['experiments/audit_snap_tier_resources.py']
assert [arm['arm'] for arm in r['rows']] == ['snap', 'gpu', 'cpu_scores']
assert len(r['teacher_token_ids']) == 50
summaries = []
for arm in r['rows']:
    assert len(arm['boundaries']) == 51 and len(arm['predicted_token_ids']) == 50
    for step, row in enumerate(arm['boundaries']):
        assert row['step'] == step and len(row['layers']) == 32
        assert [l['layer'] for l in row['layers']] == list(range(32))
        if arm['arm'] == 'snap':
            assert all(l['visible_rows'] == 974 + step and l['stored_visible_heads'] == 32 for l in row['layers'])
            expected = (974 + step) * 32 * 32 * 128 * 2 * 2
            assert row['visible_kv_tensor_payload_bytes'] == expected
            assert row['distinct_owned_kv_storage_bytes'] == {'cuda:0': expected}
        else:
            hot = cold = reserved = 0
            for l in row['layers']:
                assert l['hot_rows'] == 1024 and l['visible_rows'] == 1056 and l['stored_visible_heads'] == 8
                assert 0 <= l['shadow_rows'] <= l['cold_rows'] <= 2048
                assert l['hot_payload_bytes'] == l['hot_rows'] * 8 * 128 * 2 * 2
                assert l['cold_occupied_payload_bytes'] == l['cold_rows'] * 8 * 128 * 2 * 2
                hot += l['hot_payload_bytes']; cold += l['cold_occupied_payload_bytes']; reserved += l['cold_reserved_payload_bytes']
            visible = 32 * 1056 * 8 * 128 * 2 * 2
            assert row['visible_kv_tensor_payload_bytes'] == visible
            if arm['arm'] == 'gpu':
                assert cold == reserved
                assert row['distinct_owned_kv_storage_bytes'] == {'cuda:0': hot + cold + visible}
            else:
                assert reserved == 32 * (2048 + 1024 + 1) * 8 * 128 * 2 * 2
                assert row['distinct_owned_kv_storage_bytes'] == {'cuda:0': hot + visible, 'cpu': reserved}
        assert row['process_gpu_allocated_bytes'] >= row['distinct_owned_kv_storage_bytes']['cuda:0']
        assert row['process_gpu_reserved_bytes'] >= row['process_gpu_allocated_bytes']
    boundaries = arm['boundaries']
    summaries.append(dict(arm=arm['arm'],
        owned_gpu_kv_storage_range_bytes=[min(b['distinct_owned_kv_storage_bytes']['cuda:0'] for b in boundaries), max(b['distinct_owned_kv_storage_bytes']['cuda:0'] for b in boundaries)],
        owned_cpu_kv_storage_range_bytes=[min(b['distinct_owned_kv_storage_bytes'].get('cpu',0) for b in boundaries), max(b['distinct_owned_kv_storage_bytes'].get('cpu',0) for b in boundaries)],
        boundary_visible_rows_range=[min(l['visible_rows'] for b in boundaries for l in b['layers']), max(l['visible_rows'] for b in boundaries for l in b['layers'])],
        actual_decode_attended_rows_range=[min(l['visible_rows']+1 for b in boundaries[:-1] for l in b['layers']), max(l['visible_rows']+1 for b in boundaries[:-1] for l in b['layers'])],
        final_distinct_owned_kv_storage_bytes=boundaries[-1]['distinct_owned_kv_storage_bytes']))
gpu, cpu = r['rows'][1:]
assert gpu['predicted_token_ids'] == cpu['predicted_token_ids']
for g, c in zip(gpu['boundaries'], cpu['boundaries']):
    for gl, cl in zip(g['layers'], c['layers']):
        assert {k:v for k,v in gl.items() if k != 'cold_reserved_payload_bytes'} == {k:v for k,v in cl.items() if k != 'cold_reserved_payload_bytes'}
assert all(a['boundaries'][-1]['distinct_owned_kv_storage_bytes'] != b['boundaries'][-1]['distinct_owned_kv_storage_bytes'] for a,b in zip(r['rows'],r['rows'][1:]))
with tarfile.open(bundle,'w:gz') as archive:
    for path in list(r['source_sha256']) + ['results/baseline_fidelity/verify_snap_tier_resource_trajectory_20261001.py']:
        archive.add(path,arcname=path)
report = dict(identity_geometry_storage_equations_verified=True, raw_sha256=sha(raw), source_bundle_sha256=sha(bundle),
    input_token_sha256=hashlib.sha256(json.dumps(r['input_token_ids'],separators=(',',':')).encode()).hexdigest(),
    boundaries_per_arm=51, layers=32, arms=summaries, resource_matched=False,
    gpu_cpu_geometry_and_predictions_equal=True, final_gpu_cpu_cold_occupied_payload_bytes=sum(l['cold_occupied_payload_bytes'] for l in gpu['boundaries'][-1]['layers']),
    limitation='Snapshot KV storage includes visible cache copy and physical host reservation, excluding metadata/model/logits. Callback GPU synchronizations invalidate latency claims. Pinned allocator retained blocks/process RSS, equal-byte/width quality and physical STT/controller gates remain open.')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
