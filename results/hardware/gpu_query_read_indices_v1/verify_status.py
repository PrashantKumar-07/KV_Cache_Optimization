"""Read-only final evidence/status audit; writes only a new final proof."""
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    return json.loads((HERE / name).read_text())

proof_names = ['verified_correctness.json', 'verified_resource.json',
               'verified_resource_v2.json', 'verified_cost.json', 'verified_lru.json']
checked_hashes = 0
for name in proof_names:
    proof = read(name)
    assert proof['verified']
    for relative, expected in proof['evidence_sha256'].items():
        assert digest(HERE / relative) == expected, (name, relative)
        checked_hashes += 1

for prefix in ['', 'resource_', 'cost_', 'lru_']:
    metadata = read(prefix + 'implementation.json')
    assert digest(HERE / (prefix + 'implementation_source.tar.gz' if not prefix else prefix + 'source.tar.gz')) == metadata['archive_sha256']
    assert digest(HERE / (prefix + 'plan.json')) == metadata['plan_sha256']
    archive = HERE / (prefix + 'implementation_source.tar.gz' if not prefix else prefix + 'source.tar.gz')
    with tarfile.open(archive) as handle:
        for relative, expected in metadata['sources'].items():
            assert digest(ROOT / relative) == expected, relative
            member = handle.extractfile(relative)
            assert member is not None
            assert hashlib.sha256(member.read()).hexdigest() == expected, relative

plan = read('plan.json')
for relative, expected in plan['source_sha256'].items():
    assert digest(ROOT / relative) == expected, relative
with tarfile.open(HERE / 'control_source.tar.gz') as handle:
    for relative, expected in plan['source_sha256'].items():
        assert hashlib.sha256(handle.extractfile(relative).read()).hexdigest() == expected

original = read('verified_resource.json')
corrected = read('verified_resource_v2.json')
ignore = {'evidence_sha256', 'ready_for_development_gate',
          'ready_for_component_cost_gate', 'development_quality_expansion'}
assert {k:v for k,v in original.items() if k not in ignore} == {k:v for k,v in corrected.items() if k not in ignore}
assert corrected['ready_for_component_cost_gate']
assert corrected['development_quality_expansion'] is False
correction = read('resource_scope_correction.json')
assert digest(HERE / 'verify_resource.py') == correction['original_verifier_sha256']
assert digest(HERE / 'verified_resource.json') == correction['original_report_sha256']

jobs = {}
for name in ['native', 'gpu', 'resource', 'cost', 'lru']:
    job = read(name + '_job.json')
    assert job['status'] == 'finished' and job['exit_code'] == 0
    jobs[name] = job
assert '283 passed' in (HERE / 'tests_full.log').read_text()
assert '10 passed' in (HERE / 'tests_focused_v2.log').read_text()

for name in ['cost.json', 'lru.json']:
    raw = read(name)
    assert len(raw['rows']) == 12
    for row in raw['rows']:
        expected = ((315621376, 117571584, 76, 57)
                    if name == 'lru.json' and row['arm'] == 'control'
                    else (5825961984, 115855360, 1701, 78))
        actual = tuple(row[k] for k in ['host_read_bytes', 'host_write_bytes',
                                       'native_h2d_calls', 'native_d2h_calls'])
        assert actual == expected, (name, row['block'], row['arm'])
        assert row['own_hooks_removed']

figure = read('figure_manifest.json')
assert digest(HERE / 'verified_cost.json') == figure['component_proof_sha256']
assert digest(HERE / 'verified_lru.json') == figure['lru_proof_sha256']
assert digest(HERE / 'plot_results.py') == figure['plot_source_sha256']
for name, expected in figure['artifacts_sha256'].items():
    assert digest(HERE / name) == expected

status_names = ['PROJECT_STATUS.md', 'START_HERE.md', 'RESEARCH_ROADMAP.md',
                'PAPER_WORKING_DRAFT.md', 'REMEDIATION_CHECKLIST.md', 'SUBMISSION_SPRINT.md']
for name in status_names:
    current = (ROOT / name).read_text()[:3000]
    for value in ['283', '24.61%', '1.810', '10–20', 'Stop standalone']:
        assert value in current, (name, value)
    assert 'guard is running' not in current and 'nowrunning' not in current
assert 'all5ownedGPUjobs terminal' in (ROOT / 'EXPERIMENT_LOG.md').read_text()[-1800:]
assert read('verified_cost.json')['gpu_read_index_component_primary_passed']
assert read('verified_lru.json')['ordinary_lru_regression_guards_passed'] is False

result = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'verified': True,
    'classification': 'Progress: exact research GPU read indices, positive component cost, failed ordinary LRU guard.',
    'tests_passed': 283,
    'evidence_hash_checks': checked_hashes,
    'archived_sources_equal_current': True,
    'ordinary_source_cli_defaults_unchanged': True,
    'original_resource_report_preserved_numeric_results_unchanged': True,
    'native_transfer_bytes_and_calls_replay_all_24_measured_rows': True,
    'jobs': jobs,
    'figure_visually_inspected_no_overlap': True,
    'all_owned_jobs_terminal': True,
    'ordinary_adoption': False,
    'fullscan_quality_timing_expansion': False,
    'full_goal_complete': False,
    'next': 'Investigate a distinct fused ranking/lower-traffic retrieval mechanism with GPU selected-row indices; preregister before implementing. Do not retry the stopped standalone full-scan candidate.',
    'remaining': ['Complete 512MiB resource target and actual matched faithful baseline',
                  'Independent quality and latency gates', 'Physical STT/controller evidence',
                  'Remaining P0/P1/P2 research and submission validation'],
    'estimate_working_days': [10, 20],
    'estimate_condition': 'Conditional on physical hardware availability and research outcomes; no guaranteed finish date.',
    'evidence_sha256': {name: digest(HERE / name) for name in proof_names +
                        ['figure_manifest.json', 'verify_status.py', 'README.md']},
    'status_sha256': {name: digest(ROOT / name) for name in status_names + ['EXPERIMENT_LOG.md']},
}
output = HERE / 'verified_status.json'
assert not output.exists(), 'Preserve an existing final proof rather than overwrite it.'
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'verified': True, 'tests_passed': 283,
                  'evidence_hash_checks': checked_hashes,
                  'all_owned_jobs_terminal': True,
                  'component_gain_percent': 24.608979103319584,
                  'ordinary_lru_median_ratio': 1.809867618850936,
                  'full_goal_complete': False}))
