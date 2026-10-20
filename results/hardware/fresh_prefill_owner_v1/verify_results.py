"""Read-only fresh-caller portability decision and ordinary frozen replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PRIOR = ROOT / 'results/hardware/bulk_resource_audit_v1'
spec = importlib.util.spec_from_file_location('bulk_resource_verifier', PRIOR / 'verify_results.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
MIB = 2**20


def main():
    output = BASE / 'verified_report.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'plan.json'); jobs = read(BASE / 'jobs.json')
    assert sha(BASE / 'source.tar.gz') == plan['source_bundle_sha256']
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile()}
    assert sources == plan['source_sha256']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    assert sha(ROOT / 'results/hardware/prefill_owner_share_v1/verified_scoped_preservation.json') == plan['prior_proof_sha256']
    prior_proof = read(PRIOR / 'verified_report.json')
    assert sha(PRIOR / 'verified_report.json') == plan['ordinary_prior_proof_sha256']
    assert prior_proof['verified']
    old = read(PRIOR / 'fresh_shared_bulk.json')
    assert sha(PRIOR / 'fresh_shared_bulk.json') == prior_proof['evidence_sha256']['fresh_shared_bulk.json']
    data = {name: read(BASE / (name + '.json')) for name in ('ordinary', 'shared')}
    rows = {}; replay_fields = ('predicted_token_ids', 'teacher_token_ids', 'input_ids_sha256',
        'prompt_sha256', 'teacher_source_sha256', 'state_sha256', 'counters', 'occupied_host_bytes', 'reserved_host_bytes')
    for name, raw in data.items():
        assert jobs[name]['status'] == 'finished' and jobs[name]['exit_code'] == 0
        assert raw['passed'] and raw['fresh_owner_arm'] == name and raw['owner_hooks_removed']
        assert (raw['hot'], raw['cold']) == (991, 1090)
        assert len(raw['predicted_token_ids']) == len(raw['teacher_token_ids']) == 50
        assert raw['fresh_owner_driver_sha256'] == sources['results/hardware/fresh_prefill_owner_v1/audit_fresh_owner.py']
        for key in ('source_sha256', 'actual_source_sha256'):
            assert all(sources[p] == value for p, value in raw[key].items())
        for key in replay_fields:
            assert raw[key] == old[key], (name, key)
        assert raw['reserved_host_bytes'] == raw['owned_pinned_peak_bytes'] == 144.25 * MIB
        assert raw['owned_host_allocator_block_peak_bytes'] == 256 * MIB
        assert raw['owned_pinned_after_cleanup_bytes'] == raw['owned_host_allocator_block_after_cleanup_bytes'] == 0
        resource = raw['resource_model']; base = resource['constant_model_storage_bytes']
        assert sum(resource['model_ranges'].values()) == base == 14483464704
        for snapshot in ('loaded_model_snapshot', 'decode_final_snapshot'):
            module.check_snapshot(resource[snapshot], resource['model_ranges'])
        calls = raw['logits_lifetime_rows']
        assert len(calls) == 51 and calls[0]['logits_bytes'] == 1000 * MIB
        assert calls[0]['shape'] == [1, 16384, 32000]
        assert not calls[0]['prefill_logits_alive_before_forward'] and calls[1]['prefill_logits_alive_before_forward']
        assert all(not r['prefill_logits_alive_before_forward'] for r in calls[2:])
        steps = 50 if name == 'shared' else 0
        assert raw['owner_counts'] == dict(released_forwards=steps, restored_forwards=steps, released_layers=steps * 32)
        counts = raw['prototype_counts']
        assert counts['prefill_shared_layers'] == (32 if steps else 0)
        assert counts['raw_bound_layers'] == steps * 32 and 0 <= counts['raw_compactions'] <= steps * 32
        if steps:
            assert counts['raw_compactions'] > 0
        peaks = raw['gpu_phase_peaks']; host = raw['owned_host_allocator_block_peak_bytes']
        assert peaks['prefill_peak'] == peaks['whole_allocated_peak']
        rows[name] = dict(active_decode_envelope_mib=(peaks['decode_peak'] - base + host) / MIB,
            reserved_envelope_mib=(peaks['whole_reserved_peak'] - base + host) / MIB,
            prefill_peak_gb=peaks['prefill_peak'] / 1e9)
    for key in ('host_read_bytes', 'host_write_bytes', 'host_initial_write_bytes', 'native_h2d_calls', 'native_d2h_calls'):
        assert data['ordinary']['prototype_counts'][key] == data['shared']['prototype_counts'][key], key
    for key in ('prefill_peak', 'whole_allocated_peak'):
        assert data['ordinary']['gpu_phase_peaks'][key] == data['shared']['gpu_phase_peaks'][key]
    control, candidate = rows['ordinary']['active_decode_envelope_mib'], rows['shared']['active_decode_envelope_mib']
    reduction = 100 * (1 - candidate / control)
    result = dict(verified=True, fresh_caller_memory_primary_passed=reduction >= 10,
        reduction_pct=reduction, rows=rows, frozen_ordinary_replay_rows=2,
        prefill_logits_retained_mib=1000, all_transfer_payload_and_call_counts_match=True,
        complete512MiB_gate_passed=False, ordinary_candidate_ready=False,
        advance_to_usual_fresh_portability=reduction >= 10,
        decision='fresh_caller_primary_passed_additional_guards_needed' if reduction >= 10 else 'stop_fresh_caller_portability_expansion',
        evidence_sha256={p: sha(BASE / p) for p in ('plan.json', 'jobs.json', 'source.tar.gz', 'ordinary.json', 'shared.json')},
        verifier_sha256=sha(Path(__file__)),
        limits='Paired unchanged fresh caller/allocator, one model/seed/fixed teacher. The separate legacy result remains valid in its scope. No speed/quality-superiority/physicalSTT/full-resource equality or ordinary adoption claim.')
    assert control > 512 and candidate > 512
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
