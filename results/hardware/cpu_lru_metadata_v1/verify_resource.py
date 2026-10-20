"""Frozen ordinary50step resource replay with charged timestamp tensor/copies."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_resource.json';assert not output.exists()
    plan=read(BASE/'resource_plan.json');frozen=read(BASE/'resource_implementation.json');raw=read(BASE/'resource.json')
    assert sha(BASE/'resource_plan.json')==frozen['plan_sha256'] and sha(BASE/'gpu.json')==plan['gpu_sha256']
    assert sha(BASE/'resource_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'resource_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    job=read(BASE/'resource_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and raw['mirrors_released'] and raw['timestamp_residency_cpu']
    assert sha(ROOT/plan['reference'])==plan['reference_sha256'];ref=read(ROOT/plan['reference'])
    for field in ['input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids','predicted_token_ids','state_sha256','counters']:
        assert raw[field]==ref[field],field
    assert len(raw['predicted_token_ids'])==50
    for f in raw['prototype_counts']:assert raw['prototype_counts'][f]==ref['prototype_counts'][f]
    for field in ['owned_pinned_after_cleanup_bytes','owned_host_allocator_block_after_cleanup_bytes']:assert raw[field]==0
    assert raw['timestamp_refreshes']==51 and len(raw['cpu_metadata_boundary_rows'])==51
    rows=raw['cpu_metadata_boundary_rows'];assert all(r['new_owned_cpu_metadata_tensor_bytes']==r['timestamp_tensor_bytes']+r['temporary_packed_tensor_bytes'] for r in rows)
    assert raw['timestamp_h2d_bytes']==sum(r['temporary_packed_tensor_bytes'] for r in rows)==5692032
    assert raw['cpu_metadata_tensor_peak_bytes']==max(r['new_owned_cpu_metadata_tensor_bytes'] for r in rows)==229632
    assert raw['timestamp_initial_d2h_bytes']==104576
    constant=raw['resource_model']['constant_model_storage_bytes'];assert constant==ref['resource_model']['constant_model_storage_bytes']==14483464704
    host=raw['owned_host_allocator_block_peak_bytes'];assert host==ref['owned_host_allocator_block_peak_bytes']==256*2**20
    original=(ref['gpu_phase_peaks']['decode_peak']-constant+host)/2**20
    active=(raw['gpu_phase_peaks']['decode_peak']-constant+host+raw['cpu_metadata_tensor_peak_bytes'])/2**20
    passing=active<=576
    result=dict(verified=True,resource_primary_passed=passing,ordinary_control_active_envelope_mib=original,candidate_active_plus_cpu_metadata_tensor_bound_mib=active,declared_pilot_ceiling_mib=576,full512mib_passed=False,
        candidate_whole_reserved_envelope_plus_cpu_metadata_tensor_bound_mib=(raw['gpu_phase_peaks']['whole_reserved_peak']-constant+host+raw['cpu_metadata_tensor_peak_bytes'])/2**20,
        prefill_peak_gb=raw['gpu_phase_peaks']['prefill_peak']/1e9,rounded_owned_host_mib=256,cpu_metadata_tensor_peak_bytes=229632,
        timestamp_h2d_bytes=5692032,timestamp_initial_d2h_bytes=104576,timestamp_refreshes=51,
        unchanged_native_kv_h2d_bytes=315621376,unchanged_native_kv_d2h_bytes=117571584,native_h2d_calls=76,native_d2h_calls=57,
        exact_frozen50pred_state_counters_kv_replay=True,mirrors_pinned_roots_released=True,ready_for_cost_gate=passing,ordinary_adoption=False,full_goal_complete=False,
        limits=plan['limits']+' Active scope includes newlyownedCPU tensorroots but not unknown CPUmallocrounding; no completeprocess512MiB proof. No timing or independentquality claim.',
        evidence_sha256={f:sha(BASE/f) for f in ['resource_plan.json','resource_implementation.json','resource_source.tar.gz','resource.json','resource_job.json','audit_resources.py','verified_correctness.json','verify_resource.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
