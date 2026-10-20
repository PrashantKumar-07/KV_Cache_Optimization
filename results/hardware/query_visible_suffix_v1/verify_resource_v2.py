"""Check GPU read-index query resources, charged metadata and unchanged selection."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    plan=read(BASE/'resource_plan.json');frozen=read(BASE/'resource_implementation.json');d=read(BASE/'resource.json')
    assert sha(BASE/'resource_plan.json')==frozen['plan_sha256']
    assert sha(BASE/'verified_correctness.json')==plan['correctness_proof_sha256']
    assert sha(BASE/'resource_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'resource_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources']
    assert all(sha(ROOT/f)==v for f,v in sources.items())
    job=read(BASE/'resource_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert d['passed'] and (d['hot'],d['cold'])==(991,1090) and d['prefill_last_logits']
    assert d['query_hooks_removed'] and d['query_selected_layers']==1600 and d['query_scanned_rows']>0
    assert read(BASE/'resource_cleanup.json')==dict(selectors_closed=True,attention_pre_hooks_restored=True)
    assert d['owned_pinned_after_cleanup_bytes']==d['owned_host_allocator_block_after_cleanup_bytes']==0
    old=read(ROOT/plan['reference']);assert sha(ROOT/plan['reference'])==plan['reference_sha256']
    supplement=read(BASE/'resource_native_source.json')
    correction=read(BASE/'resource_native_source_correction.json')
    assert sha(BASE/'resource_native_source.tar.gz')==supplement['archive_sha256']
    assert sha(BASE/'resource.json')==supplement['measured_resource_sha256']
    assert correction['original_verifier_sha256']==sha(BASE/'verify_resource.py')
    assert correction['original_implementation_sha256']==sha(BASE/'resource_implementation.json')
    assert correction['supplement_sha256']==sha(BASE/'resource_native_source.json')
    with tarfile.open(BASE/'resource_native_source.tar.gz') as t:
        native={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert native==supplement['sources'] and not set(native)&set(sources)
    assert all(sha(ROOT/f)==v==d['actual_source_sha256'][f]==old['actual_source_sha256'][f] for f,v in native.items())
    sources.update(native)
    for field in ('input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids'):
        assert d[field]==old[field],field
    for key in ('source_sha256','actual_source_sha256'):
        assert all(sources[f]==v for f,v in d[key].items())
    const=d['resource_model']['constant_model_storage_bytes'];assert const==old['resource_model']['constant_model_storage_bytes']
    host=d['owned_host_allocator_block_peak_bytes'];assert host==old['owned_host_allocator_block_peak_bytes']==256*2**20
    assert d['reserved_host_bytes']==d['owned_pinned_peak_bytes']==(1090+64)*131072
    active=(d['gpu_phase_peaks']['decode_peak']-const+host)/2**20
    old_active=(old['gpu_phase_peaks']['decode_peak']-const+host)/2**20
    assert len(d['predicted_token_ids'])==50
    for field in ('predicted_token_ids','state_sha256','counters','prototype_counts'):assert d[field]==old[field],field
    assert d['visible_suffix_updates']==1600
    assert d['native_h2d_calls']==old['native_h2d_calls']==1701 and d['native_d2h_calls']==old['native_d2h_calls']==78
    assert d['index_metadata_released'] and d['index_refreshes']==51 and d['index_reads']==1600
    assert d['index_gpu_storage_bytes']==32*(1090+64)*(8*4+8)
    assert d['index_metadata_h2d_bytes']==51*d['index_gpu_storage_bytes']
    passing=active<=576
    result=dict(verified=True,resource_primary_passed=passing,
        control_active_decode_envelope_mib=old_active,candidate_active_decode_envelope_mib=active,
        declared_active_ceiling_mib=576,rounded_owned_host_mib=host/2**20,
        candidate_reserved_envelope_mib=(d['gpu_phase_peaks']['whole_reserved_peak']-const+host)/2**20,
        candidate_prefill_peak_gb=d['gpu_phase_peaks']['prefill_peak']/1e9,
        changed_teacher_predictions=sum(a!=b for a,b in zip(d['predicted_token_ids'],old['predicted_token_ids'])),
        selected_layer_forwards=d['query_selected_layers'],visible_suffix_updates=d['visible_suffix_updates'],scanned_cold_rows=d['query_scanned_rows'],
        control_native_h2d_bytes=old['prototype_counts']['host_read_bytes'],
        candidate_native_h2d_bytes=d['prototype_counts']['host_read_bytes'],
        candidate_native_d2h_bytes=d['prototype_counts']['host_write_bytes'],
        native_h2d_calls=d['native_h2d_calls'],native_d2h_calls=d['native_d2h_calls'],
        roots_hooks_released=True,all_owned_jobs_finished=True,full512mib_passed=active<=512,
        ready_for_component_cost_gate=passing,development_quality_expansion=False,ordinary_adoption=False,full_goal_complete=False,
        candidate_index_metadata_h2d_bytes=d['index_metadata_h2d_bytes'],candidate_index_gpu_storage_bytes=d['index_gpu_storage_bytes'],
        index_refreshes=d['index_refreshes'],index_native_reads=d['index_reads'],index_metadata_released=True,
        next='Preregister3ABBA GPU-index query concat vs suffix composition cost; ordinaryLRU guard required before adoption. No oldfullscan quality retry.' if passing else 'Stop expansion.',
        limits='Single model/seed fixed16K50 teacher audit. Equal teacher predictions do not prove free-generation quality; '
            'Large full-scan nativeKV traffic and candidate index metadata charged separately; old generic index metadata '
            'copies not independently measured. No totalH2D saving, quiet timing, actual baseline equality, physicalSTT or adoption proof.',
        evidence_sha256={f:sha(BASE/f) for f in ('resource_plan.json','resource_implementation.json','resource_source.tar.gz',
            'resource.json','resource_cleanup.json','resource_job.json','verified_correctness.json','verify_resource.py','verify_resource_v2.py','resource_native_source.json','resource_native_source.tar.gz','resource_native_source_correction.json')},
        frozen_control_sha256=sha(ROOT/plan['reference']))
    output=BASE/'verified_resource_v2.json';assert not output.exists()
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
