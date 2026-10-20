"""Fresh ownership memory primary with frozen ordinary behavior/traffic."""
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
    with tarfile.open(BASE/'resource_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    job=read(BASE/'resource_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and raw['append_leases_closed'] and read(BASE/'resource_cleanup.json')['append_leases_closed']
    assert sha(ROOT/plan['reference'])==plan['reference_sha256'];ref=read(ROOT/plan['reference'])
    for f in ['input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids','predicted_token_ids','state_sha256','counters']:assert raw[f]==ref[f],f
    for f,v in raw['prototype_counts'].items():assert v==ref['prototype_counts'][f]
    assert len(raw['predicted_token_ids'])==50 and raw['append_fast_updates']==1600 and raw['append_fallback_updates']==0
    assert raw['reserved_buffer_allocations']==32 and raw['shared_hot_prefixes']==1632
    assert raw['shared_transition_bindings']==raw['shared_immediate_compactions']==1600
    for f in ['owned_pinned_after_cleanup_bytes','owned_host_allocator_block_after_cleanup_bytes']:assert raw[f]==0
    constant=raw['resource_model']['constant_model_storage_bytes'];assert constant==ref['resource_model']['constant_model_storage_bytes']==14483464704
    host=raw['owned_host_allocator_block_peak_bytes'];assert host==ref['owned_host_allocator_block_peak_bytes']==256*2**20
    original=(ref['gpu_phase_peaks']['decode_peak']-constant+host)/2**20
    active=(raw['gpu_phase_peaks']['decode_peak']-constant+host)/2**20
    reduction=1-active/original;prefillratio=raw['gpu_phase_peaks']['prefill_peak']/ref['gpu_phase_peaks']['prefill_peak']
    guards=dict(declared_active_512=active<=512,relative_active_10pct=reduction>=.10,prefill_peak=prefillratio<=1.10)
    passing=(guards['declared_active_512'] or guards['relative_active_10pct']) and guards['prefill_peak']
    result=dict(verified=True,resource_primary_passed=passing,guards=guards,ordinary_active_mib=original,candidate_active_mib=active,relative_active_reduction=reduction,candidate_reserved_envelope_mib=(raw['gpu_phase_peaks']['whole_reserved_peak']-constant+host)/2**20,prefill_peak_gb=raw['gpu_phase_peaks']['prefill_peak']/1e9,prefill_peak_ratio=prefillratio,owned_pinned_allocator_mib=256,
        exact_frozen50pred_state_counters_native_kv_replayed=True,fast_append_updates=1600,fallback_append_updates=0,shared_hot_prefixes=1632,reserved_buffer_allocations=32,shared_transition_bindings=1600,shared_immediate_compactions=1600,own_hooks_methods_pinned_roots_released=True,
        ready_for_wholeloop_cost_gate=passing,ordinary_adoption=False,full_goal_complete=False,
        disposition='Resourceprimary passed; preregisterquiet3ABBA wholeloop speedgate.' if passing else 'Stop unifiedsharedtransition: resourceprimaryfailed. No macro timing/quality expansion.',
        limits=plan['limits']+' Unified model/policy buffer ownership; oldstandalonestop preserved separately. No independentquality/faithfulbaselinewin or physicalSTT claim.',
        evidence_sha256={f:sha(BASE/f) for f in ['resource_plan.json','resource_implementation.json','resource_source.tar.gz','resource.json','resource_job.json','resource_cleanup.json','audit_resources.py','verified_correctness.json','verify_resource.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
