"""Verify actual resource scopes and reject unequal-use comparison claims."""
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def archive(metadata,filename):
    m=read(BASE/metadata);assert sha(BASE/filename)==m['archive_sha256']
    with tarfile.open(BASE/filename) as t:
        sources={p.name:hashlib.sha256(t.extractfile(p).read()).hexdigest() for p in t.getmembers() if p.isfile()}
    assert sources==m['sources']
    assert all(sha(ROOT/p)==v for p,v in sources.items())
    return m,sources


def main():
    output=BASE/'verified_report.json';assert not output.exists()
    plan=read(BASE/'plan.json');m,sources=archive('implementation.json','implementation_source.tar.gz')
    assert sha(BASE/'plan.json')==m['plan_sha256'] and sha(BASE/'cpu.json')==m['cpu_sha256']
    assert all(sources[p]==v for p,v in plan['control_sources_sha256'].items())
    assert all(sha(ROOT/p)==v for p,v in plan['controls_sha256'].items())
    cpu_meta,cpu_sources=archive('cpu_observation_implementation.json','cpu_observation_source.tar.gz')
    assert sha(BASE/'cpu_observation_plan.json')==cpu_meta['plan_sha256']
    cpu_plan=read(BASE/'cpu_observation_plan.json')
    assert sha(ROOT/cpu_plan['driver'])==cpu_plan['driver_sha256']
    assert sha(ROOT/cpu_plan['control'])==cpu_plan['control_sha256']
    selfcheck=read(BASE/'cpu.json');assert selfcheck['passed'] and len(selfcheck['pairs'])==3 and selfcheck['partial_error_cleanup']
    jobs=read(BASE/'jobs.json');assert len(jobs)==2 and all(j['status']=='finished' and j['exit_code']==0 for j in jobs.values())
    snap=read(BASE/'snap.json');cpu=read(BASE/'cpu_gpu.json');frozen=read(ROOT/cpu_plan['control'])
    assert snap['passed'] and cpu['passed']
    assert snap['audit_driver_sha256']==sources[str((BASE/'audit.py').relative_to(ROOT))]
    assert cpu['ordinary_width_driver_sha256']==cpu_sources[str((BASE/'audit_cpu.py').relative_to(ROOT))]
    assert snap['adapter_hooks_and_groups_restored']
    assert read(BASE/'cpu_width_cleanup.json')==dict(root_width_hook_removed=True,nested_driver_restored=True)
    assert read(BASE/'cpu_gpu_cleanup.json')==dict(root_hooks_restored=True,leases_closed=True)
    for field in ('predicted_token_ids','teacher_token_ids','input_ids_sha256','prompt_sha256','teacher_source_sha256',
                  'state_sha256','counters','prototype_counts','gpu_phase_peaks','occupied_host_bytes','reserved_host_bytes',
                  'owned_pinned_peak_bytes','owned_host_allocator_block_peak_bytes'):
        assert cpu[field]==frozen[field],field
    old=read(ROOT/'results/baseline_fidelity/snap_tier_owned_envelope_20261002/result.json')
    faithful=next(row for row in old['rows'] if row['arm']=='snap')
    assert snap['predicted_token_ids']==faithful['predicted_token_ids'] and len(snap['predicted_token_ids'])==50
    for field in ('teacher_token_ids','input_ids_sha256','prompt_sha256','teacher_source_sha256'):
        assert snap[field]==cpu[field],field
    assert snap['prefill_last_logits'] and cpu['prefill_last_logits']
    for data in (snap,cpu):
        assert data['owned_pinned_after_cleanup_bytes']==0
        assert all(sources[p]==v for p,v in data['source_sha256'].items())
        assert data['runtime']['packages']==snap['runtime']['packages']
    assert cpu['owned_host_allocator_block_after_cleanup_bytes']==0
    assert snap['owned_pinned_peak_bytes']==0
    assert snap['constant_model_storage_bytes']==cpu['resource_model']['constant_model_storage_bytes']
    baseline=snap['constant_model_storage_bytes'];assert baseline==14483464704
    assert len(snap['boundaries'])==51
    for step,row in enumerate(snap['boundaries']):
        layers=row['cache']['layers'];assert len(layers)==32
        assert all(layer['visible_rows']==974+step and layer['stored_visible_heads']==32 for layer in layers)
        assert row['cache']['distinct_owned_kv_storage_bytes']=={'cuda:0':(974+step)*512*1024}
    snap_width=snap['forward_attended_rows_by_layer']
    assert snap_width==[[975+i]*32 for i in range(50)]
    observations=cpu['ordinary_width_observations'];assert len(observations)==51
    assert observations[0]==dict(input_rows=16384,before_cache_rows=[],before_cache_heads=[])
    cpu_width=[]
    for observation in observations[1:]:
        assert observation==dict(input_rows=1,before_cache_rows=[1023]*32,before_cache_heads=[8]*32)
        cpu_width.append([n+observation['input_rows'] for n in observation['before_cache_rows']])
    different_steps=sum(x!=y for x,y in zip(cpu_width,snap_width));assert different_steps==49
    host=cpu['owned_host_allocator_block_peak_bytes'];assert host==256*2**20
    rows={}
    for arm,data,host_bytes in (('snap',snap,0),('cpu_bulk',cpu,host)):
        phase=data['gpu_phase_peaks']
        rows[arm]=dict(gpu_decode_active_mib=(phase['decode_peak']-baseline)/2**20,
            owned_pinned_allocator_capacity_mib=host_bytes/2**20,
            combined_active_decode_envelope_mib=(phase['decode_peak']-baseline+host_bytes)/2**20,
            combined_reserved_envelope_mib=(phase['whole_reserved_peak']-baseline+host_bytes)/2**20,
            whole_prefill_peak_gb=phase['prefill_peak']/1e9,
            forward_width_min=min(min(x) for x in (snap_width if arm=='snap' else cpu_width)),
            forward_width_max=max(max(x) for x in (snap_width if arm=='snap' else cpu_width)),
            stored_visible_kv_heads=32 if arm=='snap' else 8,
            explicit_h2d_kv_payload_bytes=data.get('explicit_h2d_kv_bytes',data.get('prototype_counts',{}).get('host_read_bytes',0)),
            explicit_d2h_kv_payload_bytes=data.get('explicit_d2h_kv_bytes',data.get('prototype_counts',{}).get('host_write_bytes',0)))
    assert rows['cpu_bulk']['combined_active_decode_envelope_mib']==541.0771484375
    assert rows['snap']['combined_active_decode_envelope_mib']==528.7412109375
    quality=read(ROOT/'results/baseline_fidelity/snap_tier_envelope_fresh_quality_20261002/verified_report.json')['quality']
    assert not quality['noninferiority_passed'] and not quality['superiority_passed']
    result=dict(verified=True,resources=rows,decode_steps=50,layer_forward_width_observations=3200,
        steps_with_unequal_forward_width=different_steps,
        exact_actual_byte_and_width_match=False,complete_512mib_active_envelope_passed=False,
        frozen_faithful_and_ordinary_predictions_replayed=True,all_owned_jobs_finished=True,
        ordinary_source_changed=False,matched_resource_p0_complete=False,full_goal_complete=False,
        frozen_independent_direct_cpu_quality=dict(task_balanced_difference=quality['task_balanced_mean_difference'],
            ci95=quality['paired_document_bootstrap_ci95'],noninferiority_passed=False,
            scope='Frozen older DirectCPU shared-logical-ceiling quality, not current bulk under exact matched resources; no new current quality claim.'),
        next='Preregister a comparison design that controls actual allocation ceilings and observed attention geometry before rerunning quality/latency; do not call these configurations an exact resource match.',
        limits='One-model/seed/16K50. Memory scope is PyTorch CUDA allocations minus constant model storage plus owned pinned allocator ranges; unowned host/driver allocations excluded. Reserved envelope includes retained CUDA blocks from prefill. Diagnostic times invalid; no paired latency, current independent quality, complete resource equality or physicalSTT claim.',
        evidence_sha256={p:sha(BASE/p) for p in ('plan.json','control_source.tar.gz','implementation.json','implementation_source.tar.gz',
            'cpu.json','cpu_observation_plan.json','cpu_observation_implementation.json','cpu_observation_source.tar.gz',
            'snap.json','cpu_gpu.json','cpu_gpu_cleanup.json','cpu_width_cleanup.json','jobs.json')},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))


if __name__=='__main__':main()
