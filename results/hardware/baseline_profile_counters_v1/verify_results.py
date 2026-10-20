"""Read-only narrow reporting diff, five-policy GPU schema and bulk replay."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    output = BASE / 'verified_report.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'plan.json'); implementation = read(BASE / 'implementation.json')
    assert sha(BASE / 'plan.json') == implementation['plan_sha256']
    assert sha(BASE / 'latency_profile.py.before') == plan['before_sha256']
    assert sha(BASE / 'source.tar.gz') == implementation['archive_sha256']
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            for m in archive.getmembers() if m.isfile()}
    assert sources == implementation['sources']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    old = ast.parse((BASE / 'latency_profile.py.before').read_text())
    new = ast.parse((ROOT / 'experiments/latency_profile.py').read_text())
    def counter_assignment(tree):
        return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'counters']
    a, b = counter_assignment(old), counter_assignment(new)
    assert len(a) == len(b) == 1
    b[0].value = a[0].value
    assignments = [node for node in ast.walk(new) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            and target.value.id == 'counters' for target in node.targets)]
    assert len(assignments) == 1
    count = assignments[0]
    assert ast.dump(count.value) == ast.dump(ast.parse('len(steps)', mode='eval').body)
    parents = [node for node in ast.walk(new) if isinstance(node, ast.If) and count in node.body]
    assert len(parents) == 1 and len(parents[0].body) == 1
    lists = [value for node in ast.walk(new) for _, value in ast.iter_fields(node)
        if isinstance(value, list) and parents[0] in value]
    assert len(lists) == 1
    lists[0].remove(parents[0])
    assert ast.dump(old) == ast.dump(new), 'Non-reporting AST changed'
    assert '8 failed, 2 passed' in (BASE / 'tests_before.log').read_text()
    assert '262 passed, 1 warning' in (BASE / 'tests_after.log').read_text()
    jobs = read(BASE / 'jobs.json')
    assert jobs['gpu']['status'] == 'finished' and jobs['gpu']['exit_code'] == 0
    gpu = read(BASE / 'gpu.json')
    assert gpu['passed'] and gpu['hook_cleanup_verified'] and not gpu['timing_valid']
    assert all(sources[p] == value for p, value in gpu['source_sha256'].items())
    assert len(gpu['rows']) == 10 and len(gpu['bulk_rows']) == 2
    grouped = {}; observed_forwards = 0
    for row in gpu['rows']:
        name, last = row['policy'], row['last_logits']
        pair = grouped.setdefault(name, {})
        assert last not in pair; pair[last] = row
        events = row['events']; observed_forwards += len(events)
        assert len(events) == 39
        for start in (0, 13, 26):
            assert events[start]['input_shape'] == [1, 4096]
            assert events[start]['logit_shape'] == [1, 1 if last else 4096, 32000]
            for step, event in enumerate(events[start + 1:start + 13]):
                assert event['token'] == gpu['teacher_tokens'][step]
                assert event['positions'] == [[4096 + step]]
        trials = row['result']['trials']; assert len(trials) == 2
        for trial in trials:
            assert trial['decode_steps'] == 12
            assert len(trial['step_ms']) == len(trial['predicted_token_ids']) == 12
            for key in ('paid_writes', 'writes_saved', 'demoted', 'promoted'):
                assert (key in trial) == (name == 'tiered')
                if name == 'tiered':
                    assert trial[key] >= 0
    assert set(grouped) == {'full', 'streaming', 'h2o', 'snapkv', 'tiered'}
    for pair in grouped.values():
        assert set(pair) == {False, True}
        assert [e['predicted'] for e in pair[False]['events']] == [e['predicted'] for e in pair[True]['events']]
    before, after = gpu['bulk_rows']
    assert before['arm'] == 'before' and after['arm'] == 'after'
    for field in ('state_sha256', 'history_sha256', 'every_visible_state_sha256',
        'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes', 'initial_d2h_payload_bytes',
        'occupied_host_bytes', 'reserved_host_bytes', 'native_d2h_calls', 'native_h2d_calls',
        'per_layer_initial_d2h_calls'):
        assert before[field] == after[field], field
    for a, b in zip(before['trials'], after['trials']):
        for field in ('predicted_token_ids', 'paid_writes', 'writes_saved', 'demoted', 'promoted', 'decode_steps'):
            assert a[field] == b[field], field
    result = dict(verified=True, baseline_fixed_stream_error_fixed=True,
        policy_count=5, prefill_logit_modes=2, observed_gpu_forwards=observed_forwards,
        ordinary_bulk_frozen_replay_passed=True, tests_passed=262, all_owned_jobs_finished=True,
        counter_scope='decode_steps counts measured model forwards. STT counters emitted only when exported in both policy snapshots; no invented baseline STT zeros.',
        source_change_scope='Only profiler counter reporting; whole-module AST otherwise identical.',
        speed_claim=False, fair_resource_comparison=False, full_goal_complete=False,
        evidence_sha256={p: sha(BASE / p) for p in ('plan.json', 'implementation.json', 'source.tar.gz',
            'jobs.json', 'gpu.json', 'tests_before.log', 'tests_after.log')},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
