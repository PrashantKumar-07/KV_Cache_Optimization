"""Freeze one reviewed development candidate. Run only after review is complete."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FULL = ROOT/'results/full_evaluation_20261015'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = BASE/'plan.json'
    if target.exists():
        raise FileExistsError(target)
    full = json.loads((FULL/'plan.json').read_text())
    parts = json.loads((FULL/'partitions.json').read_text())
    original = [row for row in parts['rows'] if row['original135']]
    assert len(original) == 135
    rows = []
    for row in original:
        task, index = row['task'], row['source_zero_based_index']
        controls = {}
        for arm in ('tiered_hot992', 'snapkv_faithful'):
            found = list((FULL/'records').glob(f'shard_*/{task}_{index:04d}_{arm}.json'))
            assert len(found) == 1
            record = json.loads(found[0].read_text())
            assert record['execution_complete'] and record['sample_id'] == row['sample_id']
            assert record['plan_sha256'] == sha(FULL/'plan.json')
            controls[arm] = dict(path=str(found[0].relative_to(ROOT)), sha256=sha(found[0]))
        rows.append(dict(row, controls=controls))
    tracked = [BASE/name for name in ('exposure_prior.py', 'test_exposure_prior.py',
        'run_candidate.py', 'prepare_plan.py', 'design.json', 'cpu_tests.log',
        'fresh_document_reserve.json', 'scope_amendment.json')]
    tracked += [FULL/'run_full.py']
    source = dict(full['source_sha256'])
    source.update({str(path.relative_to(ROOT)):sha(path) for path in tracked})
    for path, digest in full['source_sha256'].items():
        assert sha(ROOT/path) == digest, f'Frozen control source changed: {path}'
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        candidate='ExposureCalibratedPriorPolicy', candidate_count=135,
        rows=rows, protocol=full['protocol'], model=full['model'],
        model_identity=full['model_identity'], runtime=full['runtime'],
        source_sha256=source, dataset_files=full['dataset_files'],
        dataset_sha256=full['dataset_sha256'], full_plan_sha256=sha(FULL/'plan.json'),
        candidate_configuration=full['arm_configs']['tiered_hot992'],
        primary_gate=dict(gain_vs_old_at_least_points=1,
            paired_target_document_cluster_CI95_lower_vs_old_strictly_above=0,
            no_task_mean_loss_beyond_points=5, point_gain_vs_faithful_snap_strictly_above=0,
            bootstrap_draws=10000, bootstrap_seed=0, task_aggregation='equal six-task mean'),
        scope='Consumed historical development only. Single fixed candidate; no sweeps. No SOTA claim.',
        stopping='Any gate failure stops expansion; preserve results and never enlarge to obtain significance.',
        checkpoint='One atomic completed record per sample. Failed attempts retained separately.',
        gpu_authorization='Root coordinates launch after quiet timing. Preparing plan does not launch GPU.')
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(plan,indent=2)+'\n')
    temporary.rename(target)
    print(json.dumps(dict(rows=135, plan_sha256=sha(target)),indent=2))


if __name__ == '__main__':
    main()
