"""Verify model state parity and interval-recomputed known policy KV storage."""
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def archived(raw,suffix=''):
    frozen=json.loads((BASE/f'source{suffix}_manifest.json').read_text())
    assert sha(BASE/f'source{suffix}.tar.gz')==frozen['bundle_sha256']
    with tarfile.open(BASE/f'source{suffix}.tar.gz') as archive:
        for path,digest in raw['source_sha256'].items():
            assert digest==frozen['source_sha256'][path]
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==digest
    return frozen['source_sha256']


def main():
    stage_path=ROOT/'results/hardware/reuse_visible_native_screen_20261002/verified_report_v2.json'
    stage=json.loads(stage_path.read_text());plan=json.loads((BASE/'plan.json').read_text())
    assert stage['verified'] and stage['advance_to_model_storage_gates']
    assert sha(stage_path)==plan['stage_proof_sha256']
    assert not stage['any_new_4ms_speed_gate_passed']
    native_binary=stage['reuse_binary_sha256'];geometries={}
    for name,hot,cold,root in (('shared',991,1090,151257088),('default',1024,2048,276824064)):
        raw=json.loads((BASE/f'parity_{name}.json').read_text())
        assert raw['passed'] and len(raw['rows'])==8 and raw['reuse_binary_sha256']==native_binary
        frozen=archived(raw)
        assert raw['config']['hot']==hot and raw['config']['cold']==cold and raw['config']['decode']==12
        assert raw['config']['contexts']==[4096,8192,16384,31500] and raw['config']['blocks']==0 and raw['config']['warmup']==0
        for i,context in enumerate((4096,8192,16384,31500)):
            control,candidate=raw['rows'][2*i:2*i+2]
            assert control['context']==candidate['context']==context
            assert control['arm']=='fused_bounded' and candidate['arm']=='native_reuse'
            for field in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes',
                'explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes',
                'native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls'):
                assert control[field]==candidate[field],(name,context,field)
            assert candidate['reserved_host_bytes']==root
            assert len(candidate['every_visible_state_sha256'])==13
            for field in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):
                assert control['trials'][0][field]==candidate['trials'][0][field]
            assert candidate['reuse_counts']=={'reused_layers':384,'fallback_layers':0,'reused_compactions':384}
        geometries[name]=dict(hot=hot,cold=cold,root_request_bytes=root,contexts=4,boundary_frames_per_arm=52,exact=True)
    resource=json.loads((BASE/'resource_shared.json').read_text())
    assert resource['passed'] and resource['native_reuse_composition'] and resource['policy_preserved']
    assert resource['reuse_binary_sha256']==native_binary
    frozen=archived(resource,'_resource')
    assert resource['composition_driver_sha256']==frozen['experiments/audit_reuse_decode_lifetimes.py']
    old_dir=ROOT/'results/hardware/decode_kv_lifetimes_20261002'
    old_source=json.loads((old_dir/'source_manifest.json').read_text())['source_sha256']
    assert frozen['experiments/audit_decode_kv_lifetimes.py']==old_source['experiments/audit_decode_kv_lifetimes.py']
    prior=json.loads((old_dir/'result.json').read_text())
    for field in ('input_token_ids','teacher_token_ids','prompt_sha256','teacher_source_sha256','seed','context','decode'):
        assert resource[field]==prior[field]
    for field in ('predicted_token_ids','counters','h2d_payload_bytes','d2h_payload_bytes'):
        assert resource['result'][field]==prior['result'][field]
    boundaries=resource['result']['boundaries'];assert len(boundaries)==51
    for row,old in zip(boundaries,prior['result']['boundaries']):
        assert row['layers']==old['layers'] and row['visible_kv_tensor_payload_bytes']==old['visible_kv_tensor_payload_bytes']
        expected_gpu=263979008 if row['step']==0 else 134217728
        assert row['distinct_owned_kv_storage_bytes']=={'cuda:0':expected_gpu,'cpu':151257088}
        if row['step']:assert row['forward_attended_rows_by_layer']==old['forward_attended_rows_by_layer']
    peaks={};sample_count=0
    assert [t['step'] for t in resource['traces']]==list(range(1,51))
    for trace in resource['traces']:
        samples=trace['samples'];n=len(samples);counts=np.zeros(n+1,dtype=np.int64);deltas={}
        assert [s['event'] for s in samples]==list(range(n))
        for index,a in enumerate(trace['allocations']):
            assert a['id']==index and a['bytes']>=0
            birth=a['birth_event'];end=n if a['end_event'] is None else a['end_event'];assert 0<=birth<=end<=n
            delta=deltas.setdefault(a['device'],np.zeros(n+1,dtype=np.int64))
            delta[birth]+=a['bytes'];delta[end]-=a['bytes'];counts[birth]+=1;counts[end]-=1
        totals={d:np.cumsum(delta)[:n] for d,delta in deltas.items()};live=np.cumsum(counts)[:n]
        for i,s in enumerate(samples):
            calculated={d:int(v[i]) for d,v in totals.items() if v[i] or d in s['known_live_kv_storage_bytes']}
            assert calculated==s['known_live_kv_storage_bytes'] and sum(calculated.values())==s['known_live_kv_total_bytes']
            assert live[i]==s['live_storage_count']
            stage_name=s['stage']
            if stage_name not in peaks or s['known_live_kv_total_bytes']>peaks[stage_name]['known_live_kv_total_bytes']:
                peaks[stage_name]=dict(step=trace['step'],**s)
        assert sum(s['operation']=='native_launch' for s in samples)==1
        sample_count+=n
    assert peaks==resource['phase_peaks']
    maximum=max(s['known_live_kv_total_bytes'] for s in peaks.values())
    old_max=max(s['known_live_kv_total_bytes'] for s in prior['phase_peaks'].values())
    result=dict(verified=True,geometries=geometries,resource_boundaries=51,resource_decode_steps=50,
        interval_sums_recomputed=True,retained_transition_samples=sample_count,
        phase_peak_mib={name:s['known_live_kv_total_bytes']/2**20 for name,s in peaks.items()},
        known_policy_kv_peak_mib=maximum/2**20,previous_known_policy_kv_peak_mib=old_max/2**20,
        known_policy_kv_peak_reduction_pct=100*(1-maximum/old_max),
        post_decode_boundary_mib=(134217728+151257088)/2**20,
        observed_policy_kv_within_512mib=maximum<=512*2**20,
        advance_to_whole_model_regression_gates=maximum<=512*2**20,
        stage_speed_4ms_gate_passed=False,reuse_binary_sha256=native_binary,
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p.name!='verified_parity_resource.json'},
        verifier_sha256=sha(Path(__file__)),limitations='Same policy-only tracker/classifier and stored boundary geometry. Known liveKV lowerbound, not allforward/attention temporaries, retained allocator blocks or actual physical equality. PostdecodeGPUstorage includes full1024-row rawcapacity behindviews. Prefill remains original fullprompt path; whole-model10%memory, new speed/quality, serving and physicalSTT claims unproven. Native stream/freegeneration/interleave and timing gates mandatory beforeCLI.')
    output=BASE/'verified_parity_resource.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
