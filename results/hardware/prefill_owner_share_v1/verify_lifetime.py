"""Read-only verification of the scoped GPU lifetime gate."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    output = BASE / 'verified_lifetime.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'lifetime_plan.json')
    assert sha(BASE / 'lifetime_source.tar.gz') == plan['source_bundle_sha256']
    with tarfile.open(BASE / 'lifetime_source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile()}
    assert sources == plan['source_sha256']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    assert sha(BASE / 'verified_report.json') == plan['prior_batch_sha256']
    previous_sources = read(BASE / 'generation_plan.json')['source_sha256']
    assert all(sha(ROOT / p) == value for p, value in previous_sources.items())
    raw = read(BASE / 'lifetime.json'); jobs = read(BASE / 'lifetime_jobs.json')
    assert raw['passed'] and len(raw['checks']) == 14 and not raw.get('error')
    assert jobs['status'] == 'finished' and jobs['exit_code'] == 0
    assert all(sources[p] == value for p, value in raw['source_sha256'].items())
    assert raw['driver_sha256'] == sources['results/hardware/prefill_owner_share_v1/check_lifetime.py']
    for consumer in (0, 1):
        rows = [r for r in raw['checks'] if r['consumer'] == consumer]
        gathers = [r for r in rows if r['case'].startswith('all_bf16')]
        assert len(gathers) == 5 and {r['kept_rows'] for r in gathers} == {64, 32, 5, 0, 1}
        assert all(r['old_hot_owners_expired'] and r['exact_payload_and_untouched_tail'] for r in gathers)
        owners = {r['case']: r for r in rows if r['case'].startswith('gpu_')}
        assert len(owners) == 2
        for name, expiry in (('gpu_forward_restoration', [True] * 4),
                ('gpu_partial_exception_restoration', [True, True, False, False])):
            row = owners[name]
            assert row['exact_payload'] and row['balanced_lease'] and row['hooks_removed']
            assert row['expected_old_model_owner_expiry'] == expiry
    evidence = ('lifetime_plan.json', 'lifetime_source.tar.gz', 'lifetime.json',
                'lifetime_jobs.json', 'verified_report.json')
    result = dict(verified=True, scoped_gpu_stream_lifetime_gate_passed=True,
        bit_gather_owner_pressure_cases=10, gpu_forward_exception_cases=4,
        prototype_sources_unchanged=True, automatic_cross_stream_support=False,
        full_mistral_exception_compute_tested=False, timing_guard_pending=True,
        ordinary_candidate_ready=False, complete_resource_equality=False,
        evidence_sha256={p: sha(BASE / p) for p in evidence}, verifier_sha256=sha(Path(__file__)),
        limits='Two serial consumers with explicit caller events/record_stream. Toy DynamicCache '
            'for GPU hook exceptions; full Mistral generation already checked separately. '
            'No concurrent serving, automatic stream handoff, timing or ordinary adoption claim.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
