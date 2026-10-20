"""Read-only quiet-GPU ABBA regression guards with frozen code/predictions."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    output = BASE / 'verified_timing.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'timing_plan.json')
    assert sha(BASE / 'timing_source.tar.gz') == plan['archive_sha256']
    with tarfile.open(BASE / 'timing_source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            for m in archive.getmembers() if m.isfile()}
    assert sources == plan['sources']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    initial = read(BASE / 'implementation.json')
    primary = read(BASE / 'verified_resources_shared.json')
    assert sha(BASE / 'verified_resources_shared.json') == plan['primary_resource_proof_sha256']
    assert sha(BASE / 'implementation.json') == primary['evidence_sha256']['implementation.json']
    assert sha(BASE / 'source.tar.gz') == initial['source_archive_sha256']
    native_sources = {p: value for p, value in initial['sources'].items()
        if p.endswith(('.cpp', '.cu'))}
    assert all(sha(ROOT / p) == value for p, value in native_sources.items())
    preflight = read(BASE / 'timing_preflight.json')
    assert preflight['quiet'] and len(preflight['samples']) == 20
    assert all(not row['unexpected'] and not row['pids'] and 'error' not in row for row in preflight['samples'])
    assert preflight['samples'][-1]['monotonic'] - preflight['samples'][0]['monotonic'] >= 38
    jobs = read(BASE / 'timing_jobs.json')
    assert jobs['status'] == 'finished'
    rows = {}; evidence = {}
    for geometry, hot, cold in (('shared', 991, 1090), ('default', 1024, 2048)):
        if geometry not in jobs['jobs']:
            rows[geometry] = dict(verified=False, reason='Stopped after earlier guard failure')
            continue
        job = jobs['jobs'][geometry]
        assert job['status'] == 'finished' and job['exit_code'] == 0
        monitor = read(BASE / f'timing_monitor_{geometry}.json')
        assert job['clean_gpu'] and monitor['clean'] and monitor['samples']
        assert all(not row['unexpected'] and 'error' not in row and
            all(pid == job['pid'] for pid in row['pids']) for row in monitor['samples'])
        raw = read(BASE / f'timing_{geometry}.json')
        assert raw['passed'] and len(raw['rows']) == 12
        assert raw['actual_control'] == 'ordinary_bulk_all_prompt_logits'
        assert raw['actual_candidate'] == 'ordinary_bulk_last_prompt_logits'
        assert raw['logits_timing_driver_sha256'] == sources[str((BASE / 'compare_timing.py').relative_to(ROOT))]
        for path, value in raw['source_sha256'].items():
            assert (sources | native_sources)[path] == value
        assert raw['config'] == dict(output=str(BASE / f'timing_{geometry}.json'),
            contexts=[16384], decode=50, warmup=1, blocks=3, hot=hot, cold=cold)
        reference = raw['rows'][0]
        for number, row in enumerate(raw['rows']):
            assert row['block'] == number // 4 and row['context'] == 16384
            assert row['arm'] == ('fused_bounded', 'native_reuse', 'native_reuse', 'fused_bounded')[number % 4]
            for field in ('state_sha256', 'history_sha256', 'every_visible_state_sha256',
                'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes', 'initial_d2h_payload_bytes',
                'occupied_host_bytes', 'reserved_host_bytes', 'native_d2h_calls', 'native_h2d_calls',
                'per_layer_initial_d2h_calls'):
                assert row[field] == reference[field], (geometry, number, field)
            assert len(row['trials']) == 1
            trial = row['trials'][0]
            assert len(trial['step_ms']) == len(trial['predicted_token_ids']) == 50
            for field in ('predicted_token_ids', 'paid_writes', 'writes_saved', 'demoted', 'promoted', 'decode_steps'):
                assert trial[field] == reference['trials'][0][field]
        summary = {}; blocks = []
        for arm in ('fused_bounded', 'native_reuse'):
            trials = [trial for row in raw['rows'] if row['arm'] == arm for trial in row['trials']]
            steps = [value for trial in trials for value in trial['step_ms']]
            summary[arm] = dict(median=statistics.median(steps), p95=float(np.percentile(steps, 95)),
                ttft=statistics.median(trial['ttft_ms'] for trial in trials),
                prefill_peak=max(trial['prefill_peak_memory_gb'] for trial in trials),
                decode_peak=max(trial['decode_peak_memory_gb'] for trial in trials))
        for block in range(3):
            blocks.append({arm: statistics.median(value for row in raw['rows']
                if row['arm'] == arm and row['block'] == block for trial in row['trials']
                for value in trial['step_ms']) for arm in summary})
        checks = {name: summary['native_reuse'][name] <= factor * summary['fused_bounded'][name]
            for name, factor in plan['guards'].items()}
        rows[geometry] = dict(verified=True, summary=summary, blocks=blocks,
            regression_guards=checks, guards_passed=all(checks.values()))
        for name in (f'timing_{geometry}.json', f'timing_monitor_{geometry}.json', f'timing_screen_{geometry}.json'):
            evidence[name] = sha(BASE / name)
    passed = all(row.get('guards_passed', False) for row in rows.values())
    result = dict(verified=True, rows=rows, both_geometry_guards_passed=passed,
        memory_candidate_ready_for_ordinary_integration=passed,
        speed_claim=False, complete_resource_equality=False,
        evidence_sha256={**evidence, **{p: sha(BASE / p) for p in
            ('timing_plan.json', 'timing_jobs.json', 'timing_preflight.json', 'timing_source.tar.gz',
                'implementation.json', 'source.tar.gz')}},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
