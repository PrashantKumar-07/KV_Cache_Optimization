"""Source-matched allocation attribution, frozen replay and observer cleanup."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
MIB = 2**20


def verify_archive(metadata_name, archive_name):
    metadata = read(BASE / metadata_name)
    assert sha(BASE / archive_name) == metadata['archive_sha256']
    with tarfile.open(BASE / archive_name) as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            for m in archive.getmembers() if m.isfile()}
    assert sources == metadata['sources']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    return sources


def main():
    output = BASE / 'verified_report.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'plan.json')
    sources = verify_archive('implementation.json', 'source.tar.gz')
    model_sources = verify_archive('model_v3_implementation.json', 'model_v3_source.tar.gz')
    assert sha(BASE / 'plan.json') == read(BASE / 'implementation.json')['plan_sha256']
    v3_plan = read(BASE / 'model_v3_plan.json')
    assert sha(BASE / 'model_v3_plan.json') == read(BASE / 'model_v3_implementation.json')['plan_sha256']
    assert sha(Path(v3_plan['library_source'])) == v3_plan['library_source_sha256']
    model_plan = read(BASE / 'model_plan.json')
    assert sha(Path(model_plan['model_source'])) == model_plan['model_source_sha256']
    prior = ROOT / 'results/hardware/prefill_last_logits_integration_v1'
    assert sha(prior / 'last_shared.json') == plan['prior_resource_sha256']
    assert sha(prior / 'verified_report.json') == plan['prior_proof_sha256']
    old = read(prior / 'last_shared.json')
    jobs = read(BASE / 'jobs.json')
    assert all(job['status'] == 'finished' for job in jobs.values())
    assert jobs['line']['exit_code'] == jobs['model_v3']['exit_code'] == 0
    assert jobs['model']['exit_code'] == jobs['model_v2']['exit_code'] == 1
    assert not read(BASE / 'model_cleanup.json')['child_hooks_restored']
    mismatches = read(BASE / 'model_v2_cleanup.json')['mismatches']
    assert {m['label'] for m in mismatches} == {f'{i}.{name}' for i in range(32) for name in ('block', 'attention')}
    assert all(m['before'] == [0, 0] and m['after'] == [0, 1] for m in mismatches)
    cleanup = read(BASE / 'model_v3_cleanup.json')
    assert cleanup['child_hooks_restored'] and cleanup['hook_count'] == 384
    assert cleanup['loader_restored'] and cleanup['writer_restored'] and not cleanup['mismatches']
    assert read(BASE / 'cleanup.json') == dict(trace_restored=True, writer_restored=True, active_frames=0)
    replay = ('predicted_token_ids', 'teacher_token_ids', 'input_ids_sha256', 'prompt_sha256',
        'teacher_source_sha256', 'state_sha256', 'counters', 'prototype_counts', 'occupied_host_bytes',
        'reserved_host_bytes', 'owned_host_allocator_block_peak_bytes', 'gpu_phase_peaks')
    data = {}
    for name, frozen in (('raw.json', sources), ('model_v3.json', model_sources)):
        d = data[name] = read(BASE / name)
        assert d['passed'] and d['prefill_last_logits'] and d['owner_hooks_removed']
        for field in replay:
            assert d[field] == old[field], (name, field)
        assert len(d['predicted_token_ids']) == 50
        for key in ('source_sha256', 'actual_source_sha256'):
            assert all(frozen[p] == value for p, value in d[key].items())
        assert d['owned_pinned_after_cleanup_bytes'] == d['owned_host_allocator_block_after_cleanup_bytes'] == 0
    line = data['raw.json']; model = data['model_v3.json']
    assert line['allocation_driver_sha256'] == sources[str((BASE / 'audit.py').relative_to(ROOT))]
    assert model['model_allocation_driver_sha256'] == model_sources[str((BASE / 'audit_model_v3.py').relative_to(ROOT))]
    trace = line['allocation_trace']
    assert line['allocation_trace_counts']['importance'] == 32
    first_peak = next(e for e in trace if e['peak_allocated'] == old['gpu_phase_peaks']['prefill_peak'])
    assert first_peak['function'] == 'caller' and first_peak['line'] < 108
    importance_live_peak = max(e['allocated'] for e in trace if e['function'] == 'importance')
    events = model['model_allocation_events']
    assert len(events) == 384
    by_stage = {(e['label'], e['phase']): e for e in events}
    assert len(by_stage) == 384
    for index in range(32):
        assert by_stage[(f'{index}.attention', 'after')]['cache_unique_bytes'] == (index + 1) * 64 * MIB
    peak = old['gpu_phase_peaks']['prefill_peak']
    first_model_peak = next(e for e in events if e['peak_allocated'] == peak)
    assert first_model_peak['label'] == '31.down' and first_model_peak['phase'] == 'before'
    assert first_model_peak['input_shape'] == [1, 16384, 14336]
    assert first_model_peak['input_dtype'] == 'torch.bfloat16'
    assert peak - first_model_peak['allocated'] == 896 * MIB
    assert by_stage[('31.up', 'after')]['peak_allocated'] < peak
    result = dict(verified=True, full_prefill_peak_gb=peak / 1e9,
        peak_first_reached='Final MLP multiplication between up-projection return and down-projection call',
        retained_full_prompt_kv_mib=2048, mlp_operand_and_product_size_mib=448,
        transient_peak_minus_down_input_live_mib=896,
        query_scoring_observed_live_peak_gb=importance_live_peak / 1e9,
        query_scoring_not_origin_of_first_global_peak=True,
        frozen_prediction_state_transfer_and_phase_peak_replay_rows=2,
        diagnostic_hook_cleanup_verified=True, failed_initial_cleanup_assertions_preserved=True,
        all_owned_jobs_finished=True, ordinary_source_changed=False, full_goal_complete=False,
        next='Preregister layerwise KV placement that returns full prompt KV to prefill attention while storing only compressed future cache; prove exact preservation and >=10% whole-prefill-peak saving before expansion.',
        limits='Read-only allocation diagnosis, one model/seed/16K. Synchronization invalidates times. Python trace local snapshots can alter temporary lifetimes; independent child-hook audit and uninstrumented frozen phase peaks agree. No measured optimization, fair-resource/quality/physicalSTT claim.',
        evidence_sha256={p: sha(BASE / p) for p in ('plan.json', 'implementation.json', 'source.tar.gz',
            'model_plan.json', 'model_v3_plan.json', 'model_v3_implementation.json', 'model_v3_source.tar.gz',
            'jobs.json', 'raw.json', 'cleanup.json', 'model.json', 'model_cleanup.json',
            'model_v2.json', 'model_v2_cleanup.json', 'model_v3.json', 'model_v3_cleanup.json')},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
