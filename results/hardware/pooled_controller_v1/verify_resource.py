"""Fresh ownership memory primary with frozen ordinary behavior/traffic."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_resource.json';assert not output.exists()
    plan=read(BASE/'resource_plan.json');frozen=read(BASE/'resource_implementation.json');raw=read(BASE/'resource.json')
    assert sha(BASE/'resource_plan.json')==frozen['plan_sha256'] and sha(BASE/'verified_correctness.json')==plan['correctness_proof_sha256']
    assert sha(BASE/'resource_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'resource_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    job=read(BASE/'resource_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and raw['append_leases_closed'] and read(BASE/'resource_cleanup.json')['append_leases_closed']
    assert sha(ROOT/plan['reference'])==plan['reference_sha256'];ref=read(ROOT/plan['reference'])
    for f in ['input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids','predicted_token_ids','state_sha256','counters']:assert raw[f]==ref[f],f
    for f,v in raw['prototype_counts'].items():assert v==ref['prototype_counts'][f]
    assert len(raw['predicted_token_ids'])==50 and raw['append_fast_updates']==1600 and raw['append_fallback_updates']==0
    assert raw['reserved_buffer_allocations']==32 and raw['shared_hot_prefixes']==1632
    assert raw['shared_transition_bindings']==raw['batched_deferred_compactions']==1600
    assert raw['shared_immediate_compactions']==0 and raw['bounded_native_groups']==0 and raw['bounded_scratch_peak_bytes']==0
    assert raw['pooled_prepare_calls']==50 and raw['pooled_compact_groups']==200 and raw['pooled_owned_gpu_bytes']==417792 and raw['pooled_scratch_peak_bytes']==32536512
    assert raw['controller_plan_calls']==50 and raw['controller_fallback_plans']==0 and raw['controller_backup_calls']==1600
    assert raw['controller_rank_groups']==51 and raw['controller_address_calls']==101
    assert raw['controller_rank_h2d_bytes']==5555328 and raw['controller_cpu_streak_array_peak']==114816 and raw['controller_address_array_peak']==65536
    for f in ['owned_pinned_after_cleanup_bytes','owned_host_allocator_block_after_cleanup_bytes']:assert raw[f]==0
    constant=raw['resource_model']['constant_model_storage_bytes'];assert constant==ref['resource_model']['constant_model_storage_bytes']==14483464704
    host=raw['owned_host_allocator_block_peak_bytes'];assert host==ref['owned_host_allocator_block_peak_bytes']==256*2**20
    original=(ref['gpu_phase_peaks']['decode_peak']-constant+host)/2**20
    active=(raw['gpu_phase_peaks']['decode_peak']-constant+host)/2**20
    reduction=1-active/original;prefillratio=raw['gpu_phase_peaks']['prefill_peak']/ref['gpu_phase_peaks']['prefill_peak']
    guards=dict(declared_active_512=active<=512,relative_active_10pct=reduction>=.10,prefill_peak=prefillratio<=1.10)
    cpu_bound=(raw['controller_cpu_streak_array_peak']+raw['controller_address_array_peak'])/2**20
    guards['active_plus_CPU_array_bound_512']=active+cpu_bound<=512
    passing=guards['declared_active_512'] and guards['active_plus_CPU_array_bound_512'] and guards['prefill_peak']
    result=dict(verified=True,resource_primary_passed=passing,guards=guards,ordinary_active_mib=original,candidate_active_mib=active,relative_active_reduction=reduction,candidate_reserved_envelope_mib=(raw['gpu_phase_peaks']['whole_reserved_peak']-constant+host)/2**20,prefill_peak_gb=raw['gpu_phase_peaks']['prefill_peak']/1e9,prefill_peak_ratio=prefillratio,owned_pinned_allocator_mib=256,candidate_active_plus_cpu_array_bound_mib=active+cpu_bound,controller_cpu_array_root_bound_mib=cpu_bound,controller_rank_descriptor_h2d_bytes=5555328,controller_rank_descriptor_upload_calls=153,pooled_descriptor_h2d_bytes=raw['pooled_descriptor_h2d_bytes'],pooled_owned_gpu_bytes=raw['pooled_owned_gpu_bytes'],pooled_prepare_calls=50,pooled_compact_groups=200,controller_plan_calls=50,controller_fallback_plans=0,controller_backup_calls=1600,controller_address_calls=101,
        exact_frozen50pred_state_counters_native_kv_replayed=True,fast_append_updates=1600,fallback_append_updates=0,shared_hot_prefixes=1632,reserved_buffer_allocations=32,shared_transition_bindings=1600,shared_immediate_compactions=0,batched_deferred_compactions=1600,bounded_native_groups=0,pooled_scratch_peak_bytes=raw['pooled_scratch_peak_bytes'],own_hooks_methods_pinned_roots_released=True,
        ready_for_wholeloop_cost_gate=passing,ordinary_adoption=False,full_goal_complete=False,
        disposition='Resourceprimary passed; preregisterquiet3ABBA wholeloop speedgate.' if passing else 'Stop pooled controller: resourceprimaryfailed. No macro timing/quality expansion.',
        limits=plan['limits']+' Native CPU controller plus pooled state ownership; CPU/C++/Python allocator rounding and native container workspace are unmeasured. Descriptor work charged by timing/GPU peaks, derived copies are not physical PCIe transactions. No independentquality/faithfulbaselinewin or physicalSTT claim.',
        evidence_sha256={f:sha(BASE/f) for f in ['resource_plan.json','resource_implementation.json','resource_source.tar.gz','resource.json','resource_job.json','resource_cleanup.json','audit_resources.py','verified_correctness.json','verify_resource.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
