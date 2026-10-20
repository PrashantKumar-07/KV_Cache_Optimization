"""Independent fixed-scope resource and exact-preservation gate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
spec = importlib.util.spec_from_file_location('snapshot_verifier',
    ROOT / 'results/hardware/bulk_resource_audit_v1/verify_results.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry', choices=('shared', 'default'), required=True)
    args = parser.parse_args()
    geometry = args.geometry
    output = BASE / f'verified_resources_{geometry}.json'
    if output.exists():
        raise FileExistsError(output)
    frozen = read(BASE / 'implementation.json')
    assert sha(BASE / 'source.tar.gz') == frozen['source_archive_sha256']
    assert sha(BASE / 'plan.json') == frozen['plan_sha256']
    assert sha(BASE / 'cpu_checks.json') == frozen['cpu_proof_sha256']
    assert read(BASE / 'cpu_checks.json')['passed']
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            for m in archive.getmembers() if m.isfile()}
    assert sources == frozen['sources']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    plan = read(BASE / 'plan.json')
    for path, value in plan['sources_control'].items():
        assert sha(BASE / (Path(path).name + '.control')) == value == sha(ROOT / path)
    prior = ROOT / 'results/hardware/prefill_output_release_v1'
    assert sha(prior / 'verified_report.json') == plan['prior_proof_sha256']
    old = read(prior / 'released.json') if geometry == 'shared' else read(
        ROOT / 'results/hardware/bulk_resource_audit_v1/fresh_default_bulk.json')
    jobs = read(BASE / ('pilot_jobs.json' if geometry == 'shared' else 'jobs.json'))
    rows = {}; raw = {}
    fields = ('predicted_token_ids', 'teacher_token_ids', 'input_ids_sha256', 'prompt_sha256',
        'teacher_source_sha256', 'state_sha256', 'counters', 'occupied_host_bytes', 'reserved_host_bytes')
    capacities = (991, 1090) if geometry == 'shared' else (1024, 2048)
    requested = (capacities[1] + 64) * 131072
    host_capacity = (256 if geometry == 'shared' else 512) * 2**20
    for arm in ('all', 'last'):
        name = f'{arm}_{geometry}'
        d = raw[arm] = read(BASE / (name + '.json'))
        assert jobs[name]['status'] == 'finished' and jobs[name]['exit_code'] == 0
        assert d['passed'] and d['prefill_logits_arm'] == arm
        assert d['fresh_owner_arm'] == 'ordinary' and d['owner_hooks_removed']
        assert d['owner_counts'] == dict(released_forwards=0, restored_forwards=0, released_layers=0)
        assert d['logits_driver_sha256'] == sources[str((BASE / 'audit_resources.py').relative_to(ROOT))]
        assert d['logits_transform_sha256'] == sources[str((BASE / 'candidate.py').relative_to(ROOT))]
        for key in ('source_sha256', 'actual_source_sha256'):
            assert all(sources[p] == value for p, value in d[key].items())
        for key in fields:
            assert d[key] == old[key], (arm, key)
        assert len(d['predicted_token_ids']) == 50 and (d['hot'], d['cold']) == capacities
        assert d['reserved_host_bytes'] == d['owned_pinned_peak_bytes'] == requested
        assert d['owned_host_allocator_block_peak_bytes'] == host_capacity
        assert d['owned_pinned_after_cleanup_bytes'] == d['owned_host_allocator_block_after_cleanup_bytes'] == 0
        model = d['resource_model']; base = model['constant_model_storage_bytes']
        assert sum(model['model_ranges'].values()) == base == 14483464704
        for name in ('loaded_model_snapshot', 'decode_final_snapshot'):
            snapshot.check_snapshot(model[name], model['model_ranges'])
        calls = d['logits_lifetime_rows']
        assert len(calls) == 51
        assert calls[0]['logits_bytes'] == (1000 * 2**20 if arm == 'all' else 64000)
        assert calls[0]['shape'] == [1, 16384 if arm == 'all' else 1, 32000]
        assert all(not r['prefill_logits_alive_before_forward'] for r in calls)
        peaks = d['gpu_phase_peaks']; host = d['owned_host_allocator_block_peak_bytes']
        rows[arm] = dict(active_decode_envelope_mib=(peaks['decode_peak'] - base + host) / 2**20,
            reserved_envelope_mib=(peaks['whole_reserved_peak'] - base + host) / 2**20,
            prefill_peak_gb=peaks['prefill_peak'] / 1e9)
    assert raw['all']['prototype_counts'] == raw['last']['prototype_counts']
    for key in ('prefill_peak', 'whole_allocated_peak', 'whole_reserved_peak'):
        assert raw['last']['gpu_phase_peaks'][key] <= 1.02 * raw['all']['gpu_phase_peaks'][key]
    if geometry == 'shared':
        assert raw['all']['gpu_phase_peaks'] == old['gpu_phase_peaks']
    reduction = 100 * (1 - rows['last']['active_decode_envelope_mib'] / rows['all']['active_decode_envelope_mib'])
    result = dict(verified=True, geometry=geometry, memory_primary_passed=reduction >= 10,
        reduction_pct=reduction, rows=rows, frozen_replay_rows=2,
        all_predictions_state_counters_and_transfers_match=True,
        actual_host_bytes=host_capacity, complete512MiB_gate_passed=False,
        ordinary_integration_ready=False, timing_guards_pending=True,
        decision='Advance preservation/timing guards' if reduction >= 10 else 'Stop memory expansion',
        evidence_sha256={p: sha(BASE / p) for p in ('plan.json', 'implementation.json', 'source.tar.gz',
            'cpu_checks.json', ('pilot_jobs.json' if geometry == 'shared' else 'jobs.json'),
            f'all_{geometry}.json', f'last_{geometry}.json')}, verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
