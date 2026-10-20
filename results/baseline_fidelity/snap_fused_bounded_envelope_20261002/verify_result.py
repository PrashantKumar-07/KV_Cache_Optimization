"""Verify boundary and transient storage plus the frozen paired timing rerun."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    audit=read(BASE/'audit.json');timing=read(BASE/'timing.json');plan=read(BASE/'plan.json')
    frozen=read(BASE/'source_manifest.json')
    assert audit['passed'] and audit['policy_preserved'] and timing['passed'] and timing['cpu_fused_bounded']
    assert sha(BASE/'source.tar.gz')==frozen['bundle_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for raw in (audit,timing):
            for name,digest in raw['source_sha256'].items():
                assert digest==frozen['source_sha256'][name]
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
    old_base=ROOT/'results/baseline_fidelity/snap_tier_owned_envelope_20261002'
    old=read(old_base/'result.json');old_proof=read(old_base/'verified_report.json')
    assert old['passed'] and old_proof['shared_ceiling_passed']
    assert sha(old_base/'verified_report.json')==plan['old_resource_sha256']
    assert sha(old_base/'verified_timing.json')==plan['old_timing_sha256']
    old_hashes=read(old_base/'source_manifest.json')['source_sha256']
    for name in ('experiments/snapkv_adapter.py','sota/snapkv/snapkv/monkeypatch/snapkv_utils.py',
        'experiments/audit_snap_tier_resources.py','experiments/audit_snap_tier_envelope.py',
        'experiments/longbench_eval.py','src/tiered_kv_cache.py'):
        assert frozen['source_sha256'][name]==old_hashes[name],name
    stress=ROOT/'results/hardware/fused_bounded_pool_20261002/verified_stress.json'
    assert read(stress)['verified'] and sha(stress)==plan['stress_sha256']
    assert audit['binary_sha256']==timing['fused_binary_sha256']
    for field in ('seed','context','decode'):
        assert audit[field]==timing[field]==plan[field]
    for field in ('input_token_ids','teacher_token_ids','prompt_sha256','teacher_source_sha256','settings'):
        assert audit[field]==timing[field]==old[field],field
    current=audit['result'];reference=next(r for r in old['rows'] if r['arm']=='cpu_direct')
    assert len(current['boundaries'])==len(reference['boundaries'])==51
    for field in ('predicted_token_ids','counters','h2d_payload_bytes','d2h_payload_bytes'):
        assert current[field]==reference[field],field
    boundary_sizes=[]
    for row,previous in zip(current['boundaries'],reference['boundaries']):
        assert row['step']==previous['step']
        assert row['distinct_owned_kv_storage_bytes']=={'cuda:0':263979008,'cpu':151257088}
        assert sum(row['distinct_owned_kv_storage_bytes'].values())==396*2**20
        assert row['visible_kv_tensor_payload_bytes']==previous['visible_kv_tensor_payload_bytes']
        for layer,old_layer in zip(row['layers'],previous['layers']):
            assert layer['cold_reserved_payload_bytes']==1154*4096
            assert {k:v for k,v in layer.items() if k!='cold_reserved_payload_bytes'}=={
                k:v for k,v in old_layer.items() if k!='cold_reserved_payload_bytes'}
        if row['step']:
            assert row['forward_attended_rows_by_layer']==previous['forward_attended_rows_by_layer']
            assert len(row['forward_attended_rows_by_layer'])==32 and max(row['forward_attended_rows_by_layer'])<=1024
        boundary_sizes.append(sum(row['distinct_owned_kv_storage_bytes'].values()))
    launches=audit['gather_launches']
    assert [r['step'] for r in launches]==list(range(1,51))
    for launch in launches:
        roles=launch['role_storage_bytes']
        assert roles['pinned_pool']=={'cpu':151257088}
        assert roles['raw_visible']=={'cuda:0':1024*131072}
        assert roles['hot_inputs']=={'cuda:0':sum(launch['source_rows'])*4096}
        assert roles['fused_output']=={'cuda:0':991*131072}
        assert launch['kept_rows']==991 and len(launch['source_rows'])==32
        assert launch['output_kv_bytes']==991*131072
        assert launch['output_metadata_bytes']==2*32*991*4
        # These roles have disjoint storage in this recorded homogeneous path.
        role_sum=sum(v for role in roles.values() for v in role.values())
        assert role_sum==sum(launch['known_live_kv_storage_bytes'].values())==launch['known_live_kv_total_bytes']
    transient=max(r['known_live_kv_total_bytes'] for r in launches)
    assert transient==audit['maximum_known_gather_kv_bytes']==553910272
    assert audit['known_gather_storage_within_512mib']==(transient<=512*2**20)
    assert len(timing['rows'])==12 and [r['arm'] for r in timing['rows']]==plan['order']
    controls={'cpu_direct':current,'snap':next(r for r in old['rows'] if r['arm']=='snap')}
    for index,row in enumerate(timing['rows']):
        assert row['block']==index//4 and len(row['step_ms'])==len(row['forward_ms'])==50
        control=controls[row['arm']]
        for field in ('predicted_token_ids','counters','h2d_payload_bytes','d2h_payload_bytes'):
            assert row[field]==control[field]
        for field in ('layers','distinct_owned_kv_storage_bytes','visible_kv_tensor_payload_bytes'):
            assert row['final_boundary'][field]==control['boundaries'][-1][field]
    summary={}
    for arm in ('snap','cpu_direct'):
        rows=[r for r in timing['rows'] if r['arm']==arm]
        steps=[v for r in rows for v in r['step_ms']];assert len(steps)==300
        summary[arm]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)),
            ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
            prefill_gpu_peak_bytes=max(r['prefill_peak_gpu_bytes'] for r in rows),
            decode_gpu_peak_bytes=max(r['decode_peak_gpu_bytes'] for r in rows),
            final_owned_kv_bytes=rows[0]['final_boundary']['distinct_owned_kv_storage_bytes'],
            block_median_ms=[statistics.median(v for r in rows if r['block']==b for v in r['step_ms']) for b in range(3)])
    a,b=summary['snap'],summary['cpu_direct']
    gates=dict(median_gain_5pct=b['median_ms']<=.95*a['median_ms'],p95_within_2pct=b['p95_ms']<=1.02*a['p95_ms'],
        all_blocks_faster=all(x<y for x,y in zip(b['block_median_ms'],a['block_median_ms'])),
        ttft_within_10pct=b['ttft_ms']<=1.10*a['ttft_ms'])
    result=dict(verified=True,source_data_teacher_repeatability_verified=True,summary=summary,
        performance_gates=gates,all_performance_gates_passed=all(gates.values()),
        cpu_over_snap_decode_ratio=b['median_ms']/a['median_ms'],
        cpu_vs_snap_ttft_change_pct=100*(b['ttft_ms']/a['ttft_ms']-1),
        boundary_max_mib=max(boundary_sizes)/2**20,known_gather_peak_mib=transient/2**20,
        transient_ceiling_passed=transient<=512*2**20,actual_resources_equal=False,
        boundary_frames=51,gather_launch_observations=50,forward_layer_width_observations=1600,
        prior_quality_gate_rerun=False,shader_and_retention_sources_unchanged=True,
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p.name!='verified_report.json'},
        source_bundle_sha256=sha(BASE/'source.tar.gz'),verifier_sha256=sha(Path(__file__)),
        limitations='Known gather storage is a lower bound, excludes metadata/other temporaries/allocator retained blocks. Same boundary ceilings do not bound transient memory or imply actual equality. Frozen one-model/seed/teacher trajectory, not quality, serving, physicalSTT or model-wide memory superiority.')
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
