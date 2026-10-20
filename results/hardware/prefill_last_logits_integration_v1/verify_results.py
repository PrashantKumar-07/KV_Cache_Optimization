"""Verify real ordinary CLI/API paths against frozen, guarded prototypes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PRIOR = ROOT / 'results/hardware/prefill_last_logits_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
spec = importlib.util.spec_from_file_location('snapshot_verifier',
    ROOT / 'results/hardware/bulk_resource_audit_v1/verify_results.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)
FIELDS = ('prediction', 'generated_token_ids', 'score', 'generation', 'max_gen',
    'termination', 'prompt_sha256', 'state_sha256', 'trajectory_sha256',
    'trajectory_frames', 'counters', 'prefill_query_shapes', 'explicit_h2d_payload_bytes',
    'explicit_d2h_payload_bytes', 'occupied_host_tensor_bytes', 'reserved_host_tensor_bytes')


def main():
    output = BASE / 'verified_report.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'plan.json'); initial = read(BASE / 'implementation.json')
    final = read(BASE / 'validation_implementation.json')
    assert sha(BASE / 'plan.json') == initial['plan_sha256']
    assert sha(BASE / 'evaluator_plan.json') == initial['evaluator_plan_sha256']
    assert sha(BASE / 'implementation.json') == final['initial_implementation_sha256']
    for name, metadata in (('source.tar.gz', initial), ('validation_source.tar.gz', final)):
        assert sha(BASE / name) == metadata['archive_sha256']
        with tarfile.open(BASE / name) as archive:
            sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                for m in archive.getmembers() if m.isfile()}
        assert sources == metadata['sources']
    assert all(sha(ROOT / p) == value for p, value in final['sources'].items())
    changed = {p for p in initial['sources'] if initial['sources'][p] != final['sources'][p]}
    assert changed <= {'tests/test_prefill_output_lifetime.py'}
    assert changed == set(final['changes'])
    assert '252 passed, 1 warning' in (BASE / 'tests_verified.log').read_text()
    assert sha(BASE / 'cli_help.json') == initial['cli_help_sha256']
    assert read(BASE / 'cli_help.json')['passed']
    for path, value in plan['prerequisites'].items():
        assert sha(ROOT / path) == value
    timing = read(PRIOR / 'verified_timing.json')
    assert timing['verified'] and timing['both_geometry_guards_passed']
    for name, value in timing['evidence_sha256'].items():
        assert sha(PRIOR / name) == value
    prior_gen = read(PRIOR / 'verified_generation.json')
    jobs = read(BASE / 'jobs.json')
    assert len(jobs) == 6 and all(job['status'] == 'finished' and job['exit_code'] == 0 for job in jobs.values())
    replay_fields = ('predicted_token_ids', 'teacher_token_ids', 'input_ids_sha256', 'prompt_sha256',
        'teacher_source_sha256', 'state_sha256', 'counters', 'occupied_host_bytes', 'reserved_host_bytes', 'prototype_counts')
    resources = {}; generation_pairs = generation_replays = 0
    for geometry, hot, cold in (('shared', 991, 1090), ('default', 1024, 2048)):
        raw = read(BASE / f'last_{geometry}.json')
        prototype = read(PRIOR / f'last_{geometry}.json')
        proof = read(PRIOR / f'verified_resources_{geometry}.json')
        assert proof['verified'] and proof['memory_primary_passed']
        assert sha(PRIOR / f'last_{geometry}.json') == proof['evidence_sha256'][f'last_{geometry}.json']
        assert raw['passed'] and raw['prefill_last_logits'] and raw['owner_hooks_removed']
        assert (raw['hot'], raw['cold']) == (hot, cold)
        for field in replay_fields:
            assert raw[field] == prototype[field], (geometry, field)
        for key in ('source_sha256', 'actual_source_sha256'):
            assert all(final['sources'][p] == value for p, value in raw[key].items())
        assert raw['owned_pinned_after_cleanup_bytes'] == raw['owned_host_allocator_block_after_cleanup_bytes'] == 0
        model = raw['resource_model']; base = model['constant_model_storage_bytes']
        assert sum(model['model_ranges'].values()) == base == 14483464704
        for name in ('loaded_model_snapshot', 'decode_final_snapshot'):
            snapshot.check_snapshot(model[name], model['model_ranges'])
        calls = raw['logits_lifetime_rows']
        assert len(calls) == 51 and calls[0]['shape'] == [1, 1, 32000] and calls[0]['logits_bytes'] == 64000
        assert all(not row['prefill_logits_alive_before_forward'] for row in calls)
        peaks = raw['gpu_phase_peaks']; host = raw['owned_host_allocator_block_peak_bytes']
        assert host == (256 if geometry == 'shared' else 512) * 2**20
        active = (peaks['decode_peak'] - base + host) / 2**20
        assert abs(active - proof['rows']['last']['active_decode_envelope_mib']) <= 1
        assert peaks['prefill_peak'] == prototype['gpu_phase_peaks']['prefill_peak']
        resources[geometry] = dict(active_decode_envelope_mib=active,
            reserved_envelope_mib=(peaks['whole_reserved_peak'] - base + host) / 2**20,
            prefill_peak_gb=peaks['prefill_peak'] / 1e9)
        name = f'generation_{geometry}.json'
        old = read(PRIOR / name); new = read(BASE / name)
        assert sha(PRIOR / name) == prior_gen['evidence_sha256'][name]
        assert new['passed'] and len(new['rows']) == 24 and not new['native_checks']
        assert (new['hot'], new['cold']) == (hot, cold)
        assert new['manifest_sha256'] == old['manifest_sha256']
        frozen = {(r['task'], r['sample_id'], r['seed'], r['arm']): r for r in old['rows']}
        groups = {}
        for row in new['rows']:
            key = row['task'], row['sample_id'], row['seed'], row['arm']
            for field in FIELDS:
                assert row[field] == frozen[key][field], (geometry, key, field)
            groups.setdefault(key[:3], set()).add(row['arm'])
            generation_replays += 1
        assert len(groups) == 12 and all(arms == {'fused_bounded', 'native_reuse'} for arms in groups.values())
        generation_pairs += len(groups)
    all_cli, last_cli = read(BASE / 'cli_all.json'), read(BASE / 'cli_last.json')
    assert all_cli['config']['prefill_last_logits'] is False and last_cli['config']['prefill_last_logits'] is True
    config_all = all_cli['config'].copy(); config_last = last_cli['config'].copy()
    config_all.pop('prefill_last_logits'); config_last.pop('prefill_last_logits')
    assert config_all == config_last
    a = all_cli['results']['qasper']['tieredkv']; b = last_cli['results']['qasper']['tieredkv']
    assert a['score'] == b['score'] and a['sample_scores'] == b['sample_scores'] and a['n'] == b['n'] == 1
    excluded = {'elapsed_s', 'prefill_s', 'prefill_peak_alloc_bytes'}
    assert {k: v for k, v in a['samples'][0].items() if k not in excluded} == {
        k: v for k, v in b['samples'][0].items() if k not in excluded}
    frozen_sample = next(row for row in read(PRIOR / 'generation_shared.json')['rows']
        if row['sample_id'] == a['samples'][0]['sample_id'])
    assert a['samples'][0]['prediction'] == frozen_sample['prediction']
    assert a['samples'][0]['score'] == frozen_sample['score']
    result = dict(verified=True, ordinary_opt_in_adopted=True, option='--prefill-last-logits',
        default_enabled=False, tests_passed=252, actual_cli_pair_passed=True,
        actual_generator_pairs=generation_pairs, frozen_replay_rows=generation_replays,
        resources=resources, both_prototype_resource_and_timing_guards_passed=True,
        all_owned_jobs_finished=True, complete512MiB_gate_passed=False,
        speed_claim=False, full_goal_complete=False,
        limits='One supported local Mistral, reused development data/one seed. Actual ordinary paths replay frozen guarded prototypes; no new ordinary speed/whole-model-peak/quality-superiority/physicalSTT/concurrent-serving claim.',
        evidence_sha256={p: sha(BASE / p) for p in ('plan.json', 'evaluator_plan.json', 'implementation.json',
            'validation_implementation.json', 'source.tar.gz', 'validation_source.tar.gz', 'jobs.json',
            'tests_verified.log', 'cli_help.json', 'cli_all.json', 'cli_last.json', 'last_shared.json',
            'last_default.json', 'generation_shared.json', 'generation_default.json')},
        verifier_sha256=sha(Path(__file__)))
    assert generation_pairs == 24 and generation_replays == 48
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
