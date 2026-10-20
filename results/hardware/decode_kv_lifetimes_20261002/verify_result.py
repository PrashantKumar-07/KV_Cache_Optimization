"""Recompute weak-storage lifetime sums and phase maxima from frozen evidence."""
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
    manifest=json.loads((BASE/'source_manifest.json').read_text())
    prior_path=ROOT/'results/baseline_fidelity/snap_fused_bounded_envelope_20261002/verified_report.json'
    assert raw['passed'] and raw['policy_preserved'] and all(raw['self_check'].values())
    assert sha(prior_path)==plan['prior_sha256']
    prior=json.loads((prior_path.parent/'audit.json').read_text())
    assert prior['passed']
    assert sha(BASE/'source.tar.gz')==manifest['bundle_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for name,digest in raw['source_sha256'].items():
            assert digest==manifest['source_sha256'][name]
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
    for name,digest in prior['source_sha256'].items():
        assert digest==manifest['source_sha256'][name],name
    assert raw['binary_sha256']==prior['binary_sha256']
    for field in ('model','seed','context','decode','input_token_ids','teacher_token_ids','prompt_sha256','teacher_source_sha256'):
        assert raw[field]==prior[field],field
    for field in ('predicted_token_ids','counters','h2d_payload_bytes','d2h_payload_bytes'):
        assert raw['result'][field]==prior['result'][field]
    assert len(raw['result']['boundaries'])==51
    for row,old in zip(raw['result']['boundaries'],prior['result']['boundaries']):
        for field in ('step','layers','distinct_owned_kv_storage_bytes','visible_kv_tensor_payload_bytes'):
            assert row[field]==old[field],field
        if row['step']:assert row['forward_attended_rows_by_layer']==old['forward_attended_rows_by_layer']
    assert [t['step'] for t in raw['traces']]==list(range(1,51))
    peaks={};visible_returns=[];dispatch_events=0;storages=0
    for trace in raw['traces']:
        samples=trace['samples'];allocations=trace['allocations'];n=len(samples)
        assert [s['event'] for s in samples]==list(range(n))
        delta={};counts=np.zeros(n+1,dtype=np.int64)
        for index,a in enumerate(allocations):
            assert a['id']==index and a['bytes']>=0
            birth=a['birth_event'];end=n if a['end_event'] is None else a['end_event']
            assert 0<=birth<=end<=n
            assert a['dtype'] in ('torch.bfloat16','torch.float32') and a['shape'][-1]==128
            device=a['device'];delta.setdefault(device,np.zeros(n+1,dtype=np.int64))
            delta[device][birth]+=a['bytes'];delta[device][end]-=a['bytes']
            counts[birth]+=1;counts[end]-=1
        totals={d:np.cumsum(values)[:n] for d,values in delta.items()}
        count=np.cumsum(counts)[:n]
        for i,s in enumerate(samples):
            calculated={d:int(values[i]) for d,values in totals.items() if values[i] or d in s['known_live_kv_storage_bytes']}
            assert calculated==s['known_live_kv_storage_bytes'],(trace['step'],i)
            assert sum(calculated.values())==s['known_live_kv_total_bytes']
            assert count[i]==s['live_storage_count']
            key=s['stage']
            if key not in peaks or s['known_live_kv_total_bytes']>peaks[key]['known_live_kv_total_bytes']:
                peaks[key]=dict(step=trace['step'],**s)
        assert sum(s['operation']=='native_launch' for s in samples)==1
        returned=[s for s in samples if s['operation']=='visible_return']
        assert len(returned)==1
        visible_returns.append(returned[0]['known_live_kv_total_bytes'])
        dispatch_events+=n;storages+=len(allocations)
    assert peaks==raw['phase_peaks']
    result=dict(verified=True,source_data_policy_and_boundaries_preserved=True,
        exact_lifetime_interval_sums_recomputed=True,decode_steps=50,boundary_frames=51,
        dispatch_samples=dispatch_events,storage_lifetimes=storages,
        phase_peak_mib={name:r['known_live_kv_total_bytes']/2**20 for name,r in peaks.items()},
        visible_return_range_mib=[min(visible_returns)/2**20,max(visible_returns)/2**20],
        independent_512mib_violations=[name for name,r in peaks.items() if r['known_live_kv_total_bytes']>512*2**20],
        compaction_only_change_cannot_close_all_observed_violations=True,
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p.name!='verified_report.json'},
        source_bundle_sha256=sha(BASE/'source.tar.gz'),verifier_sha256=sha(Path(__file__)),
        limitations=raw['limitations'])
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
