"""Recompute frozen SnapKV known storage lifetimes and compare observed scopes."""
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raw=json.loads((BASE/'result.json').read_text())
    plan=json.loads((BASE/'plan.json').read_text())
    frozen=json.loads((BASE/'source_manifest.json').read_text())
    cpu_dir=ROOT/'results/hardware/decode_kv_lifetimes_20261002'
    cpu=json.loads((cpu_dir/'verified_report.json').read_text())
    assert cpu['verified'] and sha(cpu_dir/'verified_report.json')==plan['cpu_proof_sha256']
    assert raw['passed'] and raw['predictions_and_boundaries_preserved'] and all(raw['self_check'].values())
    assert sha(BASE/'source.tar.gz')==frozen['bundle_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for name,digest in raw['source_sha256'].items():
            assert digest==frozen['source_sha256'][name]==hashlib.sha256(archive.extractfile(name).read()).hexdigest()
    cpu_source=json.loads((cpu_dir/'source_manifest.json').read_text())['source_sha256']
    assert frozen['source_sha256']['experiments/audit_decode_kv_lifetimes.py']==cpu_source['experiments/audit_decode_kv_lifetimes.py']
    prior=json.loads((ROOT/'results/baseline_fidelity/snap_tier_owned_envelope_20261002/result.json').read_text())
    assert prior['passed']
    for key in ('input_token_ids','teacher_token_ids','prompt_sha256','teacher_source_sha256'):
        assert raw[key]==prior[key]
    reference=next(r for r in prior['rows'] if r['arm']=='snap')
    assert raw['predicted_token_ids']==reference['predicted_token_ids'] and len(raw['boundaries'])==51
    for row,old in zip(raw['boundaries'],reference['boundaries']):
        for key in ('step','layers','distinct_owned_kv_storage_bytes','visible_kv_tensor_payload_bytes'):
            assert row[key]==old[key]
    peaks={};samples_count=0;observed_count=0
    assert [t['step'] for t in raw['traces']]==list(range(1,51))
    for trace in raw['traces']:
        samples=trace['samples'];n=len(samples)
        assert [s['event'] for s in samples]==list(range(n))
        assert trace['observed_samples']>=n
        delta=np.zeros(n+1,dtype=np.int64);counts=delta.copy()
        for index,a in enumerate(trace['allocations']):
            assert a['id']==index and a['device']=='cuda:0' and a['bytes']>=0
            birth=a['birth_event'];end=n if a['end_event'] is None else a['end_event']
            assert 0<=birth<=end<=n
            delta[birth]+=a['bytes'];delta[end]-=a['bytes']
            counts[birth]+=1;counts[end]-=1
        amounts=np.cumsum(delta)[:n];live=np.cumsum(counts)[:n]
        for i,s in enumerate(samples):
            assert s['known_live_kv_storage_bytes']=={'cuda:0':int(amounts[i])}
            assert s['known_live_kv_total_bytes']==amounts[i] and s['live_storage_count']==live[i]
            stage=s['stage']
            if stage not in peaks or s['known_live_kv_total_bytes']>peaks[stage]['known_live_kv_total_bytes']:
                peaks[stage]=dict(step=trace['step'],**s)
        assert sum(s['operation']=='phase_enter' and s['stage']=='cache_update' for s in samples)==32
        samples_count+=n;observed_count+=trace['observed_samples']
    assert peaks==raw['phase_peaks']
    result=dict(verified=True,source_data_boundary_and_prediction_identity_verified=True,
        lifetime_interval_sums_recomputed=True,decode_steps=50,boundary_frames=51,
        retained_transition_samples=samples_count,observed_dispatch_samples=observed_count,
        phase_peak_mib={stage:s['known_live_kv_total_bytes']/2**20 for stage,s in peaks.items()},
        cpu_policy_phase_peak_mib=cpu['phase_peak_mib'],
        both_observed_paths_exceed_512mib=True,exact_physical_resources_equal=False,
        interpretation='SnapKV grows each layer in place; observed overlap is about one old channel, not a full old modelcache. CPU rawtuple/replacement and shared-hot lifetimes create several independent overlaps. Scope/classifier differences preclude an exact total-resource ratio.',
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p.name!='verified_report.json'},
        source_bundle_sha256=sha(BASE/'source.tar.gz'),cpu_proof_sha256=sha(cpu_dir/'verified_report.json'),
        verifier_sha256=sha(Path(__file__)),limitations=raw['limitations'])
    assert max(result['phase_peak_mib'].values())>512 and max(cpu['phase_peak_mib'].values())>512
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
