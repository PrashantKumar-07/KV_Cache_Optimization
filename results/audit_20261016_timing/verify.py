"""Independent strict verification of complete five-method context timing."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def positive(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and value > 0


def close(a, b, tolerance=1e-8):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance), (a, b)


def validate_raw(plan, report, vocab_size):
    """Validate completeness and raw observations before any final aggregate."""
    assert report.get('execution_complete') is True, 'Incomplete timing execution'
    assert 'error' not in report, 'Failed timing execution'
    contexts = [512, 1024, 2048, 4096, 8192, 16384]
    arms = ['full', 'streamingllm', 'h2o_adapted', 'snapkv_adapted', 'tiered_hot1024']
    assert plan['contexts'] == contexts and plan['arms'] == arms
    protocol = plan['protocol']
    for key, value in [('gpu', 1), ('seed', 0), ('decode_steps', 50), ('blocks', 3),
                       ('warmup_per_row', 1), ('repeat_per_row', 1),
                       ('measured_forwards_per_arm_context', 150)]:
        assert protocol[key] == value
    assert protocol['dtype'] == 'BF16'
    assert len(report['rows']) == 90
    observations = report['quiet_observations']
    assert len(observations) == 181
    assert observations[0]['stage'] == 'before_model_load'
    uuid = observations[0]['uuid']
    assert uuid.startswith('GPU-')
    for i, observation in enumerate(observations):
        assert observation['quiet'] is True and observation['other_compute_processes'] == []
        assert observation['gpu'] == 1 and observation['uuid'] == uuid
        if i:
            assert observation['stage'] == ('before_row' if i % 2 else 'after_row')
    expected = []
    for context in contexts:
        for block in range(3):
            order = arms[block % 5:] + arms[:block % 5]
            if block % 2:
                order = order[::-1]
            expected.extend((context, block, arm) for arm in order)
    groups = {}
    for identity, row in zip(expected, report['rows']):
        assert (row['context'], row['block'], row['arm']) == identity
        assert row['hooks_restored'] is True and row['all_logits_finite'] is True
        assert row['timing_scope'] == 'end_to_end_model_policy_token'
        assert len(row['trials']) == 1
        trial = row['trials'][0]
        assert trial['decode_steps'] == 50
        assert len(trial['step_ms']) == len(trial['forward_ms']) == len(trial['predicted_token_ids']) == 50
        for value in trial['step_ms'] + trial['forward_ms']:
            positive(value)
        for key in ['ttft_ms', 'prefill_forward_ms', 'prefill_peak_memory_gb', 'decode_peak_memory_gb']:
            positive(trial[key])
        assert trial['ttft_ms'] >= trial['prefill_forward_ms']
        assert all(step >= forward for step, forward in zip(trial['step_ms'], trial['forward_ms']))
        assert all(isinstance(t, int) and 0 <= t < vocab_size for t in trial['predicted_token_ids'])
        for key in ['ttft_ms', 'itl_ms', 'mean_itl_ms', 'forward_ms', 'peak_memory_gb', 'prefill_peak_memory_gb']:
            positive(row[key])
        close(row['ttft_ms'], round(trial['ttft_ms'], 3))
        close(row['itl_ms'], round(float(np.median(trial['step_ms'])), 3))
        close(row['mean_itl_ms'], float(np.mean(trial['step_ms'])))
        close(row['forward_ms'], float(np.median(trial['forward_ms'])))
        close(row['peak_memory_gb'], trial['decode_peak_memory_gb'])
        close(row['prefill_peak_memory_gb'], trial['prefill_peak_memory_gb'])
        if row['arm'] == 'tiered_hot1024':
            assert trial['demoted'] == trial['paid_writes'] + trial['writes_saved']
            assert all(trial[k] >= 0 for k in ['demoted', 'paid_writes', 'writes_saved', 'promoted'])
        key = (row['context'], row['arm'])
        groups.setdefault(key, []).append(trial)
    assert len(groups) == 30
    for trials in groups.values():
        assert len(trials) == 3
        assert sum(len(t['step_ms']) for t in trials) == 150
        assert all(t['predicted_token_ids'] == trials[0]['predicted_token_ids'] for t in trials)
    return groups


def self_test():
    plan = {'contexts': [512, 1024, 2048, 4096, 8192, 16384],
        'arms': ['full', 'streamingllm', 'h2o_adapted', 'snapkv_adapted', 'tiered_hot1024'],
        'protocol': {'gpu': 1, 'seed': 0, 'dtype': 'BF16', 'decode_steps': 50, 'blocks': 3,
            'warmup_per_row': 1, 'repeat_per_row': 1, 'measured_forwards_per_arm_context': 150}}
    report = {'execution_complete': True, 'rows': [], 'quiet_observations': [
        {'stage': 'before_model_load', 'gpu': 1, 'uuid': 'GPU-unit', 'quiet': True, 'other_compute_processes': []}]}
    for context in plan['contexts']:
        for block in range(3):
            order = plan['arms'][block:] + plan['arms'][:block]
            if block % 2:
                order = order[::-1]
            for arm in order:
                trial = {'decode_steps': 50, 'step_ms': [2.] * 50, 'forward_ms': [1.] * 50,
                    'predicted_token_ids': list(range(50)), 'ttft_ms': 3., 'prefill_forward_ms': 2.,
                    'prefill_peak_memory_gb': 1., 'decode_peak_memory_gb': 1.,
                    'demoted': 2, 'paid_writes': 1, 'writes_saved': 1, 'promoted': 1}
                report['rows'].append({'context': context, 'block': block, 'arm': arm,
                    'hooks_restored': True, 'all_logits_finite': True,
                    'timing_scope': 'end_to_end_model_policy_token', 'trials': [trial],
                    'ttft_ms': 3., 'itl_ms': 2., 'mean_itl_ms': 2., 'forward_ms': 1.,
                    'peak_memory_gb': 1., 'prefill_peak_memory_gb': 1.})
                for stage in ['before_row', 'after_row']:
                    report['quiet_observations'].append({'stage': stage, 'gpu': 1,
                        'uuid': 'GPU-unit', 'quiet': True, 'other_compute_processes': []})
    assert len(validate_raw(plan, report, 64)) == 30
    mutations = [lambda r: r.update(execution_complete=False),
        lambda r: r['rows'].pop(),
        lambda r: r['quiet_observations'][1].update(quiet=False),
        lambda r: r['quiet_observations'][2].update(other_compute_processes=['foreign process']),
        lambda r: r['rows'][-1]['trials'][0]['predicted_token_ids'].__setitem__(0, 63),
        lambda r: r['rows'][0]['trials'][0]['step_ms'].__setitem__(0, float('nan'))]
    for mutate in mutations:
        bad = copy.deepcopy(report)
        mutate(bad)
        try:
            validate_raw(plan, bad, 64)
        except AssertionError:
            pass
        else:
            raise AssertionError('Malformed/incomplete timing accepted')
    print(json.dumps({'synthetic_unit_checks': 'passed', 'malformed_cases_rejected': len(mutations),
                      'actual_timing_verified': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    plan_path = BASE / 'plan.json'
    raw_path = BASE / 'timing.json'
    plan = json.loads(plan_path.read_text())
    report = json.loads(raw_path.read_text())
    if not report.get('execution_complete') or len(report.get('rows', [])) != 90:
        raise SystemExit('Refusing timing verification/aggregation before complete90-row execution')
    output = BASE / 'verified_timing.json'
    assert not output.exists(), 'Preserve prior verification'
    from provenance import atomic_json, model_artifact_identity, runtime_identity
    from transformers import AutoConfig, AutoTokenizer
    assert report['plan_sha256'] == sha(plan_path)
    assert sha(BASE / 'run.py') == plan['runner_sha256']
    assert all(sha(ROOT / p) == h for p, h in plan['source_sha256'].items())
    assert runtime_identity() == plan['runtime']
    assert model_artifact_identity(plan['model']) == plan['model_identity']
    assert all(sha(ROOT / p) == h for p, h in plan['inputs_sha256'].items())
    launch = json.loads((BASE / 'launch.json').read_text())
    assert launch['gpu'] == 1 and launch['plan_sha256'] == sha(plan_path)
    binding_path = BASE / 'device_binding_observation.json'
    binding = json.loads(binding_path.read_text())
    assert binding['physical_gpu1_binding_confirmed'] is True
    assert binding['launch_pid'] == launch['pid']
    assert len(binding['matching_compute_process_rows']) == 1
    assert binding['matching_compute_process_rows'][0].split(',')[0].strip() == report['quiet_observations'][0]['uuid']
    assert int(binding['matching_compute_process_rows'][0].split(',')[1]) == launch['pid']
    config = AutoConfig.from_pretrained(plan['model'], local_files_only=True)
    groups = validate_raw(plan, report, config.vocab_size)
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    ids = tokenizer((ROOT / plan['prompt']).read_text(), return_tensors='np').input_ids
    assert ids.shape[1] >= max(plan['contexts'])
    teacher = json.loads((ROOT / plan['teachers']).read_text())['config']['teacher_token_ids']
    assert len(teacher) >= 51 and all(isinstance(t, int) and 0 <= t < config.vocab_size for t in teacher[:51])
    spec = importlib.util.spec_from_file_location('frozen_timing_policies', ROOT / 'results/full_evaluation_20261015/run_full.py')
    full = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full)
    for arm in plan['arms']:
        policy = full.policy_for(arm)
        if arm == 'streamingllm':
            assert policy.start_size == 4 and policy.recent_size == 1020
        elif arm in ('h2o_adapted', 'snapkv_adapted'):
            assert policy.budget == 1024 and policy.start_size == 4 and policy.recent_size == 128
        elif arm == 'tiered_hot1024':
            assert (policy.sram_budget, policy.stt_budget, policy.stt_expose_quota, policy.start_size,
                    policy.recent_size, policy.sttram_bifurcation_frac) == (1024, 2048, 32, 4, 128, .75)
            assert str(policy.kv_dtype) == 'torch.bfloat16'
        policy.reset()
    calls = [n for n in ast.walk(ast.parse((BASE / 'run.py').read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'profile_one']
    assert len(calls) == 1
    literals = {k.arg: k.value.value for k in calls[0].keywords if isinstance(k.value, ast.Constant)}
    assert all(literals[k] == v for k, v in {'n_decode': 50, 'n_warmup': 1,
        'n_repeat': 1, 'device': 'cuda', 'prefill_last_logits': True}.items())
    summaries = {}
    for context in plan['contexts']:
        summaries[str(context)] = {}
        for arm in plan['arms']:
            trials = groups[(context, arm)]
            steps = [x for t in trials for x in t['step_ms']]
            summaries[str(context)][arm] = {'measured_steps': len(steps), 'measured_rows': 3,
                'median_itl_ms': float(np.median(steps)), 'p95_itl_ms': float(np.percentile(steps, 95)),
                'mean_itl_ms': float(np.mean(steps)),
                'median_ttft_ms': float(np.median([t['ttft_ms'] for t in trials])),
                'max_decode_process_torch_allocated_gb': max(t['decode_peak_memory_gb'] for t in trials),
                'max_prefill_process_torch_allocated_gb': max(t['prefill_peak_memory_gb'] for t in trials)}
    result = {'verified': True, 'execution_complete': True, 'plan_sha256': sha(plan_path),
        'raw_sha256': sha(raw_path), 'verifier_sha256': sha(__file__),
        'device_binding_observation_sha256': sha(binding_path), 'rows': 90,
        'quiet_observations': 181, 'measured_decode_forwards': 4500,
        'summaries': summaries,
        'context_input_ids_sha256': {str(c): hashlib.sha256(ids[:, :c].tobytes()).hexdigest() for c in plan['contexts']},
        'teacher51_int64_sha256': hashlib.sha256(np.asarray(teacher[:51], dtype=np.int64).tobytes()).hexdigest(),
        'backend_contract': {'full': 'sdpa', 'streamingllm': 'sdpa',
            'h2o_adapted': 'eager', 'snapkv_adapted': 'eager', 'tiered_hot1024': 'eager', 'prefill_all': 'sdpa'},
        'scope': plan['protocol']['scope'], 'resource_scope': plan['protocol']['resource_scope'],
        'limitations': ['Descriptive five-method context scaling; no SOTA or advancement claim.',
            'Fixed teacher inputs ignore EOS and are not deployed greedy token throughput.',
            'Common finite-logit checks and nested synchronization add instrumentation overhead to all arms.',
            'All-logits-finite and hooks-restored flags are runtime assertions; this verifier does not replay GPU inference.',
            'Budget and backend constructor/source bindings verified; timing records do not contain complete per-layer KV geometry snapshots.',
            'Predefined rotated/reversed three-block order is not exact five-position Latin balance.',
            'GPU quiet observations sample row boundaries; CPU, thermal state and transient contention are not fully isolated.',
            'Ordinary GPU Tiered simulation; extra cold capacity and attended width disclosed; not equal total-memory or native CPU timing.']}
    atomic_json(output, result)
    print(json.dumps({'verified': True, 'rows': 90, 'quiet_observations': 181,
                      'measured_forwards': 4500, 'output': str(output)}))


if __name__ == '__main__':
    main()
