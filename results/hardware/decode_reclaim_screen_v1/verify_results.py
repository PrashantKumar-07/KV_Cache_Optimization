"""Read-only decision audit for charged per-token cache release."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output = BASE / 'verified_report.json'
    if output.exists(): raise FileExistsError(output)
    plan = json.loads((BASE / 'plan.json').read_text())
    raw = json.loads((BASE / 'pilot.json').read_text())
    assert sha(BASE / 'source.tar.gz') == plan['source_bundle_sha256']
    for name, key in (('allocator_layout_screen_20261002/verified_pilot.json', 'control_proof_sha256'),
                      ('bulk_resource_audit_v1/verified_report.json', 'latest_resource_proof_sha256')):
        assert sha(BASE.parent / name) == plan[key]
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile()}
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    assert all(sources[p] == value for p, value in raw['source_sha256'].items())
    assert raw['passed'] and raw['only_charged_decode_cache_release_differs']
    assert raw['config']['context'] == 16384 and raw['config']['decode'] == 12
    assert (raw['config']['hot'], raw['config']['cold']) == (991, 1090)
    control, candidate = raw['rows']
    assert control['arm'] == 'control' and candidate['arm'] == 'cleanup'
    for key in ('state_sha256', 'history_sha256', 'every_visible_state_sha256',
                'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes', 'initial_d2h_payload_bytes',
                'occupied_host_bytes', 'reserved_host_bytes', 'native_d2h_calls', 'native_h2d_calls'):
        assert control[key] == candidate[key], key
    assert len(control['every_visible_state_sha256']) == 13
    for key in ('predicted_token_ids', 'paid_writes', 'writes_saved', 'demoted', 'promoted', 'decode_steps'):
        assert control['trials'][0][key] == candidate['trials'][0][key], key
    for row in raw['rows']:
        assert len(row['trials']) == 1 and len(row['trials'][0]['predicted_token_ids']) == 12
        assert row['decode_reclaim']['source_sha256'] == sources['experiments/screen_decode_reclaim.py']
        assert row['decode_reclaim']['setting'] == 'expandable_segments:True'
        assert any(s['is_expandable'] for s in row['decode_reclaim']['segments'])
        resources = row['whole_decode_resources']
        assert resources['root']['allocator_block_bytes'] == resources['root']['requested_storage_bytes'] == 151257088
        assert resources['reserved_extra_plus_host_range_bytes'] == 544988672 > 512*2**20
        assert resources['allocated_extra_plus_host_range_bytes'] == 447885824
        assert row['decode_reclaim']['snapshot_totals']['reserved_bytes'] == sum(s['total_size'] for s in row['decode_reclaim']['segments'])
    before = control['whole_decode_resources']['gpu_reserved_peak_bytes']
    after = candidate['whole_decode_resources']['gpu_reserved_peak_bytes']
    gain = (before-after)/before*100
    assert gain == 0
    assert candidate['decode_reclaim']['snapshot_totals']['fully_inactive_segment_bytes'] == 0
    assert control['decode_reclaim']['snapshot_totals']['fully_inactive_segment_bytes'] == 100*2**20
    boundary_gain = (control['decode_reclaim']['snapshot_totals']['reserved_bytes'] -
                     candidate['decode_reclaim']['snapshot_totals']['reserved_bytes']) / 2**20
    assert '206 passed' in (BASE / 'full_suite.txt').read_text()
    proof = dict(verified=True, decision='reject', primary_passed=False,
                 peak_reserved_reduction_pct=gain, final_boundary_reserved_reduction_mib=boundary_gain,
                 allocated_extra_plus_host_range_mib=447885824/2**20,
                 reserved_extra_plus_host_range_mib=544988672/2**20,
                 complete_resource_equality=False, advance_to_full_timing=False,
                 advance_to_generation_or_cli=False, full_suite_passed=206,
                 evidence_sha256={n:sha(BASE/n) for n in ('plan.json','pilot.json','source.tar.gz','pilot.log','full_suite.txt')},
                 verifier_sha256=sha(Path(__file__)),
                 limitations='One paired research caller pilot, fixed reused teacher stream/model/seed. Final boundary saving is not peak saving. Hash recording invalidates timing claims; no macro timing launched after failed primary. Excludes other host/cache/driver/non-Torch resource scopes. No ordinary/default change, quality, speed, or physical STT claim.')
    output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ('verified','decision','peak_reserved_reduction_pct','final_boundary_reserved_reduction_mib','full_suite_passed')}, indent=2))


if __name__ == '__main__': main()
