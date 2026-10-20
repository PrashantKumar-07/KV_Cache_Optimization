"""Read-only source-matched generation, lease cleanup and frozen replay proof."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
FIELDS = ('prediction', 'generated_token_ids', 'score', 'generation', 'max_gen',
    'termination', 'prompt_sha256', 'state_sha256', 'trajectory_sha256',
    'trajectory_frames', 'counters', 'prefill_query_shapes',
    'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes',
    'occupied_host_tensor_bytes', 'reserved_host_tensor_bytes')


def main():
    output = BASE / 'verified_generation.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'generation_plan.json')
    assert sha(BASE / 'generation_source.tar.gz') == plan['source_bundle_sha256']
    with tarfile.open(BASE / 'generation_source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile()}
    assert sources == plan['source_sha256']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    assert sha(BASE / 'verified_pilot.json') == plan['pilot_proof_sha256']
    assert read(BASE / 'verified_pilot.json')['primary_passed']
    prior = ROOT / 'results/hardware/promotion_planning_screen_20261002'
    previous = read(prior / 'verified_integrity.json')
    assert previous['verified']
    summary = {}; replayed = 0
    for name, hot, cold in (('shared', 991, 1090), ('default', 1024, 2048)):
        filename = 'generation_' + name + '.json'
        raw = read(BASE / filename)
        assert raw['passed'] and len(raw['rows']) == 24 and not raw['native_checks']
        assert (raw['hot'], raw['cold']) == (hot, cold)
        assert raw['actual_control'] == 'ordinary_bulk_promotion'
        assert raw['actual_candidate'] == 'research_prefill_owner_share'
        assert raw['driver_sha256'] == sources['experiments/check_prefill_owner_generation.py']
        assert all(sources[p] == value for p, value in raw['source_sha256'].items())
        manifest = plan['generation_manifests'][name]
        assert raw['manifest_sha256'] == manifest['sha256'] == sha(ROOT / manifest['path'])
        assert sha(prior / filename) == previous['evidence_sha256'][filename]
        old = read(prior / filename)
        assert raw['reuse_binary_sha256'] == old['reuse_binary_sha256']
        frozen = {(r['task'], r['sample_id']): r for r in old['rows']
                  if r['arm'] == 'native_reuse'}
        pairs = {}; forwards = layers = 0
        for row in raw['rows']:
            key = row['task'], row['sample_id']
            pair = pairs.setdefault(key, {})
            assert row['arm'] not in pair and row['seed'] == 0
            pair[row['arm']] = row
            assert row['owner_hooks_removed']
            assert row['trajectory_frames'] == len(row['generated_token_ids'])
            assert row['counters']['decode_steps'] == row['trajectory_frames'] - 1
            assert row['reserved_host_tensor_bytes'] == (cold + 64) * 131072
            for field in FIELDS:
                assert row[field] == frozen[key][field], (name, key, field)
            replayed += 1
            steps = row['counters']['decode_steps'] if row['arm'] == 'native_reuse' else 0
            assert row['forward_owner_counts'] == dict(released_forwards=steps,
                restored_forwards=steps, released_layers=steps * 32)
            if row['arm'] == 'native_reuse':
                assert row['reuse_counts'] == dict(prefill_shared_layers=32,
                    raw_bound_layers=steps * 32,
                    raw_compactions=frozen[key]['reuse_counts']['reused_compactions'])
                forwards += steps; layers += steps * 32
            else:
                assert row['reuse_counts'] == {}
        assert len(pairs) == len(frozen) == 12
        assert len({k[0] for k in pairs}) == 6
        for key, pair in pairs.items():
            assert set(pair) == {'fused_bounded', 'native_reuse'}
            for field in FIELDS:
                assert pair['fused_bounded'][field] == pair['native_reuse'][field]
        summary[name] = dict(exact_generation_pairs=12, tasks=6,
            restored_forwards=forwards, restored_layers=layers)
    evidence = ('generation_plan.json', 'generation_source.tar.gz',
                'generation_shared.json', 'generation_default.json', 'verified_pilot.json')
    result = dict(verified=True, geometries=summary, frozen_generation_replay_rows=replayed,
        ordinary_candidate_ready=False, timing_guard_pending=True,
        evidence_sha256={p: sha(BASE / p) for p in evidence}, verifier_sha256=sha(Path(__file__)),
        limitations='Untimed serialized preservation on reused development data, one seed/model. '
            'No new native stream/lifetime proof, concurrency, quality superiority, complete '
            'resource equality or ordinary adoption. Further mandatory gates remain.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
