"""Verify frozen resource observations without running a model or changing raw data."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
MIB = 2**20
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda name: json.loads((BASE / name).read_text())


def check_snapshot(value, ranges):
    segments = value['segments']
    totals = value['totals']
    assert sum(s['total_size'] for s in segments) == totals['reserved_bytes']
    assert sum(s['allocated_size'] for s in segments) == totals['allocated_bytes']
    associated = exclusive = 0
    for segment in segments:
        blocks = segment['blocks']
        assert sum(b['size'] for b in blocks) == segment['total_size']
        active = [b for b in blocks if b['state'].startswith('active')]
        tagged = [any(b['address'] <= int(p) < b['address'] + b['size'] for p in ranges) for b in active]
        inactive = sum(b['size'] for b in blocks if b['state'] == 'inactive')
        if any(tagged): associated += inactive
        if tagged and all(tagged): exclusive += inactive
    assert associated == totals['inactive_in_model_segments_bytes']
    assert exclusive == totals['inactive_in_model_only_segments_bytes']


def main():
    output = BASE / 'verified_report.json'
    if output.exists(): raise FileExistsError(output)
    plan, jobs = read('plan.json'), read('jobs.json')
    assert sha(BASE / 'source.tar.gz') == plan['source_bundle_sha256']
    cli = ROOT / 'results/hardware/promotion_planning_screen_20261002/cli/verified_report.json'
    assert sha(cli) == plan['cli_proof_sha256']
    assert len(jobs) == 6 and all(j['exit_code'] == 0 for j in jobs)
    names = {j['name'] for j in jobs}
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile()}
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    data = {name: read(name + '.json') for name in names}
    for name, raw in data.items():
        assert raw['passed'] and raw['gpu'] == 'NVIDIA L40S'
        assert raw['resource_driver_sha256'] == sources['experiments/audit_bulk_resources.py']
        for key in ('source_sha256', 'actual_source_sha256'):
            assert all(sources[p] == value for p, value in raw.get(key, {}).items())
        model = raw['resource_model']
        assert sum(model['model_ranges'].values()) == model['constant_model_storage_bytes'] == 14483464704
        check_snapshot(model['loaded_model_snapshot'], model['model_ranges'])
        if 'decode_final_snapshot' in model: check_snapshot(model['decode_final_snapshot'], model['model_ranges'])
    rows = []
    for geometry, hot, cold, host in (('shared', 991, 1090, 256*MIB), ('default', 1024, 2048, 512*MIB)):
        control, bulk = (data[f'fresh_{geometry}_{arm}'] for arm in ('control', 'bulk'))
        for key in ('predicted_token_ids', 'teacher_token_ids', 'input_ids_sha256', 'state_sha256',
                    'counters', 'occupied_host_bytes', 'reserved_host_bytes', 'gpu_phase_peaks'):
            assert control[key] == bulk[key], (geometry, key)
        for arm, raw in (('control', control), ('bulk', bulk)):
            assert (raw['hot'], raw['cold']) == (hot, cold) and raw['actual_resource_arm'] == arm
            assert len(raw['predicted_token_ids']) == len(raw['teacher_token_ids']) == 50
            assert raw['reserved_host_bytes'] == (cold + 64)*131072
            assert raw['owned_host_allocator_block_peak_bytes'] == host
            assert raw['owned_pinned_after_cleanup_bytes'] == raw['owned_host_allocator_block_after_cleanup_bytes'] == 0
            calls = raw['logits_lifetime_rows']
            assert len(calls) == 51 and calls[0]['logits_bytes'] == 1000*MIB
            assert calls[0]['shape'] == [1, 16384, 32000]
            assert not calls[0]['prefill_logits_alive_before_forward']
            assert calls[1]['prefill_logits_alive_before_forward']
            assert all(not r['prefill_logits_alive_before_forward'] for r in calls[2:])
            model = raw['resource_model']; base = model['constant_model_storage_bytes']
            peaks = raw['gpu_phase_peaks']
            assert peaks['prefill_peak'] == peaks['whole_allocated_peak'] == 18627830784
            active = peaks['decode_peak'] - base + host
            reserved = peaks['whole_reserved_peak'] - base + host
            assert active > 512*MIB and reserved > 512*MIB
            rows.append(dict(geometry=geometry, caller='fresh', arm=arm,
                             allocated_extra_plus_host_range_mib=active/MIB,
                             reserved_extra_plus_host_range_mib=reserved/MIB))
        paired = data[f'legacy_{geometry}']['rows']
        assert len(paired) == 2 and {r['arm'] for r in paired} == {'fused_bounded', 'native_reuse'}
        for key in ('state_sha256', 'history_sha256', 'every_visible_state_sha256',
                    'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes', 'initial_d2h_payload_bytes',
                    'occupied_host_bytes', 'reserved_host_bytes', 'native_d2h_calls', 'native_h2d_calls'):
            assert paired[0][key] == paired[1][key], (geometry, key)
        for row in paired:
            assert len(row['trials']) == 1 and len(row['every_visible_state_sha256']) == 13
            assert len(row['trials'][0]['predicted_token_ids']) == 12
            resources = row['whole_decode_resources']; base = resources['constant_model_storage_bytes']
            assert resources['root']['allocator_block_bytes'] == host
            for kind in ('allocated', 'reserved'):
                extra = resources[f'{kind}_extra_plus_host_range_bytes']
                assert extra == resources[f'gpu_{kind}_peak_bytes'] - base + host
                assert extra > 512*MIB
            rows.append(dict(geometry=geometry, caller='legacy',
                             arm='control' if row['arm'] == 'fused_bounded' else 'bulk',
                             allocated_extra_plus_host_range_mib=resources['allocated_extra_plus_host_range_bytes']/MIB,
                             reserved_extra_plus_host_range_mib=resources['reserved_extra_plus_host_range_bytes']/MIB))
    baseline = control['resource_model']['loaded_model_snapshot']['totals']
    proof = dict(verified=True, finished_processes=6, frozen_source_files=len(sources),
                 within_caller_preservation=True, fresh_gpu_peaks_equal=True,
                 resource_ceiling_mib=512, all_observed_envelopes_exceed_ceiling=True,
                 rows=rows, model_only_inactive_mib=baseline['inactive_in_model_only_segments_bytes']/MIB,
                 unused_prefill_logits_retained_through_first_decode_mib=1000,
                 full_resource_gate_passed=False, advance_weight_packing_candidate=False,
                 evidence_sha256={n+'.json':sha(BASE/(n+'.json')) for n in sorted(names)} |
                     {n:sha(BASE/n) for n in ('plan.json','jobs.json','source.tar.gz')},
                 verifier_sha256=sha(Path(__file__)),
                 limitations='Independent GPU/host maxima form observed upper envelopes, not complete physical resource accounting. Excludes other host/cache/driver/non-Torch allocations. Fresh and legacy callers differ; their difference is not a causal effect estimate. Observer timings invalid. No quality, physical STT, or resource equality claim.')
    output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(verified=True, rows=rows, model_only_inactive_mib=proof['model_only_inactive_mib']), indent=2))


if __name__ == '__main__': main()
