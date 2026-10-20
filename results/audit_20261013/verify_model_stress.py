"""Independent raw paired-artifact checks, separate from the experiment driver."""
from pathlib import Path
import hashlib
import json

BASE = Path('results/hardware/cold_boundary_audit_20261013')
OUTPUT = BASE/'verified_audit_stress.json'
assert not OUTPUT.exists(), 'Preserve prior evidence'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
files = sorted(BASE.glob('stress_*_*_*.json')) + sorted(BASE.glob('boundary_*.json'))
assert len(files) == 6
cases = []
for path in files:
    data = json.loads(path.read_text())
    assert data['passed'] and data['owned_leases_closed']
    assert data['actual_control'] == 'bulk_payload_controller'
    assert data['actual_candidate'] == 'cold_boundary_controller'
    assert len(data['rows']) == 2
    control, candidate = data['rows']
    for field in ['state_sha256', 'history_sha256', 'every_visible_state_sha256',
                  'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes',
                  'initial_d2h_payload_bytes', 'native_h2d_calls', 'native_d2h_calls']:
        assert control[field] == candidate[field], (path, field)
    assert len(control['trials']) == len(candidate['trials']) == 1
    for field in ['predicted_token_ids', 'paid_writes', 'writes_saved', 'demoted',
                  'promoted', 'decode_steps']:
        assert control['trials'][0][field] == candidate['trials'][0][field], (path, field)
    assert len(candidate['trials'][0]['predicted_token_ids']) == 50
    cases.append({'file': str(path), 'sha256': sha(path),
                  'context': candidate.get('context'), 'paired_steps': 50,
                  'control_phase_counts': control['phase_counts'],
                  'candidate_phase_counts': candidate['phase_counts'],
                  'runtime': data['runtime'],
                  'source_sha256': data['source_sha256']})
path = BASE/'generation.json'
data = json.loads(path.read_text())
assert data['passed'] and data['owned_leases_closed'] and len(data['rows']) == 24
fields = ['prediction', 'generated_token_ids', 'score', 'generation', 'max_gen',
          'termination', 'prompt_sha256', 'state_sha256', 'trajectory_sha256',
          'trajectory_frames', 'counters', 'prefill_query_shapes',
          'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes',
          'occupied_host_tensor_bytes', 'reserved_host_tensor_bytes']
steps = 0
for first, second in zip(data['rows'][::2], data['rows'][1::2]):
    assert (first['task'], first['sample_id'], first['seed']) == (second['task'], second['sample_id'], second['seed'])
    for field in fields:
        assert first[field] == second[field], (first['task'], field)
    assert first['append_fallback_updates'] == second['append_fallback_updates'] == 0
    steps += len(first['generated_token_ids']) - 1
report = {'verified': True, 'passed': True, 'teacher_paired_steps': sum(row['paired_steps'] for row in cases),
          'teacher_cases': cases, 'free_generation_document_pairs': 12,
          'free_generation_decode_steps_per_arm': steps,
          'free_generation_tasks': sorted({row['task'] for row in data['rows']}),
          'generation_sha256': sha(path), 'generation_driver_sha256': sha(BASE/'check_generation.py'),
          'verifier_sha256': sha(Path(__file__)),
          'limits': 'Core-corrected-source exact preservation on reused development requests. No independent quality, full-logit full-model parity, current-source timing or faithful-baseline superiority. Later bounded-prefill/dtype and provenance review fixes do not change these single-token full-prefill paths; raw source identities are retained individually.',
          'adopted': False, 'full_project_complete': False}
OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'teacher_cases'}, indent=2))
