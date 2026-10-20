"""Freeze only the missing fraction ablations; never regenerate the .75 arm."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FULL = ROOT / 'results/full_evaluation_20261015'
sys.path.insert(0, str(ROOT / 'experiments'))
from provenance import atomic_json, model_artifact_identity, runtime_identity


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert not (BASE / 'plan.json').exists(), 'Frozen ablation plan must not be overwritten'
    previous = json.loads((FULL / 'plan.json').read_text())
    verified = json.loads((FULL / 'verified_results.json').read_text())
    assert verified['verified'] and verified['execution_complete']
    assert verified['plan_sha256'] == sha(FULL / 'plan.json')
    assert verified['verified_request_count'] == 10318
    assert sha(FULL / 'run_full.py') == previous['runner_sha256']
    assert model_artifact_identity(previous['model']) == previous['model_identity']
    assert runtime_identity() == previous['runtime']
    assert all(sha(ROOT / p) == digest for p, digest in previous['source_sha256'].items())
    tasks = previous['tasks']
    arms = {t: ['frac100'] if t == 'gov_report' else ['frac100', 'frac050', 'frac025'] for t in tasks}
    reused = {p: h for p, h in verified['record_sha256'].items() if p.endswith('_tiered_hot1024.json')}
    assert len(reused) == 1474
    assert all(sha(ROOT / p) == digest for p, digest in reused.items())
    count = sum(previous['counts'][t] * len(arms[t]) for t in tasks)
    assert count == 3822
    plan = {'schema': 1, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Complete the original figure8/15 headroom mechanism ablation with current genuine full-file runs; fixed descriptive ablation, not tuning or a new held-out/SOTA test.',
        'tasks': tasks, 'counts': previous['counts'], 'total_rows': 1474,
        'new_predictions': count, 'reused_fraction075_predictions': 1474,
        'arms_by_task': arms, 'fractions': {'frac100': 1., 'frac050': .5, 'frac025': .25},
        'fixed_policy': {'hot': 1024, 'cold': 2048, 'quota': 32, 'sink': 4,
            'recent': 128, 'dtype': 'bfloat16', 'actual_prompt_queries': True,
            'decode_selection': 'existing recency', 'native_cpu_offload': False},
        'protocol': previous['protocol'], 'model': previous['model'],
        'model_identity': previous['model_identity'], 'runtime': previous['runtime'],
        'source_sha256': previous['source_sha256'], 'dataset_files': previous['dataset_files'],
        'dataset_sha256': previous['dataset_sha256'], 'sample_ids': previous['sample_ids'],
        'runner_sha256': sha(BASE / 'run_headroom.py'),
        'preparation_sha256': sha(__file__),
        'frozen_generator_sha256': sha(FULL / 'run_full.py'),
        'reference_plan_sha256': sha(FULL / 'plan.json'),
        'reference_verification_sha256': sha(FULL / 'verified_results.json'),
        'reference_source_archive': str((FULL / 'source.tar.gz').relative_to(ROOT)),
        'reference_source_archive_sha256': sha(FULL / 'source.tar.gz'),
        'reused_reference_records_sha256': reused,
        'scheduling': 'two fixed global-row-index modulo2 shards; fixed per-row arm rotation by global index modulo number of task arms; no score-dependent scheduling or omission',
        'analysis': {'figure8': 'five QA tasks, all1174rows/fraction; equal-task mean QA-F1 and equal-task mean of per-task saved/(paid+saved)*100. Ratios computed from summed task counters, not mean per-example ratios.',
            'figure15': 'all six tasks, all1474rows/fraction, fixed fraction1 versus.75; per-task quality and write-event savings; mean scores use each task appropriate metric.',
            'zero_denominator': 'mark write-savings ratio undefined when task paid+saved is zero; do not invent a percentage',
            'scope': 'All outcomes descriptive after full benchmark exposure; do not select a new operating point, tune on these records, claim noninferiority from a small point difference, or make new SOTA/generalization claims.',
            'figure_contract': 'retain original visual style and scientific question; neutral titles until evidence available; no partial or archived values in final fresh figures',
            'timing': 'concurrent request elapsed times diagnostic only; modeled energy/cost explicitly analytical; logical events do not establish physical NVM endurance'},
        'resume': 'atomic immutable completed records; original.75 records reused by hash; failed attempts preserved and no implicit within-process retry'}
    atomic_json(BASE / 'plan.json', plan)
    atomic_json(BASE / 'freeze.json', {'plan_sha256': sha(BASE / 'plan.json'),
        'new_predictions': count, 'reused_predictions': len(reused),
        'source_archive_sha256': plan['reference_source_archive_sha256']})
    print(json.dumps({'plan_sha256': sha(BASE / 'plan.json'), 'new_predictions': count,
                      'reused_predictions': len(reused)}, indent=2))


if __name__ == '__main__':
    main()
