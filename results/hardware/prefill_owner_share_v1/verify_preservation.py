"""Read-only saturation, active-request leases and frozen boundary replay."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    output = BASE / 'verified_preservation.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'preservation_plan.json')
    assert sha(BASE / 'preservation_source.tar.gz') == plan['source_bundle_sha256']
    with tarfile.open(BASE / 'preservation_source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile()}
    assert sources == plan['source_sha256']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    assert sha(BASE / 'verified_pilot.json') == plan['pilot_proof_sha256']
    previous_dir = ROOT / 'results/hardware/promotion_planning_screen_20261002'
    previous = read(previous_dir / 'verified_integrity.json')
    assert previous['verified']
    summary = {}; evidence = ['preservation_plan.json', 'preservation_source.tar.gz']
    for name, hot, cold, expected_cases in (('shared', 991, 1090, 36), ('default', 1024, 2048, 4)):
        filename = 'boundary_' + name + '.json'; evidence.append(filename)
        boundary = read(BASE / filename)
        assert boundary['passed'] and boundary['fused'] and boundary['full_transient_saturation_observed']
        assert boundary['default_geometry'] == (name == 'default')
        assert boundary['preservation_kind'] == 'boundary'
        assert len(boundary['checks']) == expected_cases
        assert boundary['actual_candidate'] == 'research_prefill_owner_share'
        lengths = (4096,) if name == 'default' else (1, 31, 32, 33, 95, 96, 97, 256, 4096)
        assert {(r['inclusive'], r['fill'], r['prompt_tokens']) for r in boundary['checks']} == {
            (inclusive, fill, length) for inclusive in (True, False) for fill in (.75, 1.) for length in lengths}
        for row in boundary['checks']:
            assert row['exact_state_payload_counters'] and row['rejected_append_before_mutation']
            assert row['layers'] == 2 and row['decode_calls'] == 24
            assert row['transient_logical_peak'] <= row['slot_capacity']
        assert max(r['transient_logical_peak'] for r in boundary['checks']) == cold + 64
        filename = 'interleaved_' + name + '.json'; evidence.append(filename)
        raw = read(BASE / filename)
        assert raw['passed'] and raw['fused_bounded_interleaved']
        assert (raw['hot'], raw['cold']) == (hot, cold)
        assert len(raw['rows']) == len(raw['request_ownership_checks']) == 9
        assert raw['preservation_kind'] == 'interleaved'
        assert raw['actual_candidate'] == 'research_prefill_owner_share'
        assert sha(previous_dir / filename) == previous['evidence_sha256'][filename]
        frozen = {(r['mode'], r['request']): r for r in read(previous_dir / filename)['rows']}
        rows = {(r['mode'], r['request']): r for r in raw['rows']}
        assert len(rows) == 9 and rows.keys() == frozen.keys()
        for key, row in rows.items():
            assert row == frozen[key], (name, key)
        candidate = [r for r in raw['request_ownership_checks'] if r['candidate']]
        assert len(candidate) == 3 and sorted(r['steps'] for r in candidate) == [12, 12, 24]
        for row in candidate:
            steps = row['steps']
            assert row['prefill_shared_layers'] == 32
            assert row['raw_bound_layers'] == row['raw_compactions'] == steps * 32
            assert row['lease_counts'] == dict(released_forwards=steps,
                restored_forwards=steps, released_layers=steps * 32)
        for row in raw['request_ownership_checks']:
            if not row['candidate']:
                assert all(value == 0 for value in row['lease_counts'].values())
        frames = sum(len(rows[('interleaved_cpu', request)]['every_boundary_sha256'])
                     for request in ('A', 'B', 'C'))
        assert frames == 51
        for report in (boundary, raw):
            assert report['preservation_driver_sha256'] == sources[
                'results/hardware/prefill_owner_share_v1/check_preservation.py']
            assert all(sources[p] == value for p, value in report['preservation_source_sha256'].items())
            assert all(sources[p] == value for p, value in report['source_sha256'].items())
        summary[name] = dict(saturation_cases=expected_cases, recycled_requests=3,
            exact_boundary_frames=frames, frozen_replay_rows=9,
            restored_forwards=48, restored_layers=1536)
    result = dict(verified=True, geometries=summary, ordinary_candidate_ready=False,
        timing_guard_pending=True, stream_lifetime_gate_pending=True,
        evidence_sha256={p: sha(BASE / p) for p in evidence}, verifier_sha256=sha(Path(__file__)),
        limitations='Serialized default-stream preservation only. Synthetic multi-token paths '
            'do not invoke model leases. No threaded/async serving, new native-kernel proof, '
            'quality superiority, full-resource equality or ordinary adoption.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
