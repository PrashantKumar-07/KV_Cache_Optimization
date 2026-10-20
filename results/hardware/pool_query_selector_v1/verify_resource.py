"""Check scoped-query resources, charged transfers and cleanup without retuning."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    plan=read(BASE/'resource_plan.json');frozen=read(BASE/'resource_implementation.json');d=read(BASE/'resource.json')
    assert sha(BASE/'resource_plan.json')==frozen['plan_sha256']
    assert sha(BASE/'verified_correctness.json')==plan['correctness_sha256']
    assert sha(BASE/'resource_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'resource_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources']
    assert all(sha(ROOT/f)==v for f,v in sources.items())
    assert read(BASE/'resource_job.json')==dict(session_id=35999,status='finished',exit_code=0)
    assert d['passed'] and (d['hot'],d['cold'])==(991,1090) and d['prefill_last_logits']
    assert d['query_hooks_removed'] and d['query_selected_layers']==1600 and d['query_scanned_rows']>0
    assert read(BASE/'resource_cleanup.json')==dict(selectors_closed=True,attention_pre_hooks_restored=True)
    assert d['owned_pinned_after_cleanup_bytes']==d['owned_host_allocator_block_after_cleanup_bytes']==0
    old=read(ROOT/plan['frozen_ordinary_control'])
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
    passing=active<=576
    result=dict(verified=True,resource_primary_passed=passing,
        control_active_decode_envelope_mib=old_active,candidate_active_decode_envelope_mib=active,
        declared_active_ceiling_mib=576,rounded_owned_host_mib=host/2**20,
        candidate_reserved_envelope_mib=(d['gpu_phase_peaks']['whole_reserved_peak']-const+host)/2**20,
        candidate_prefill_peak_gb=d['gpu_phase_peaks']['prefill_peak']/1e9,
        changed_teacher_predictions=sum(a!=b for a,b in zip(d['predicted_token_ids'],old['predicted_token_ids'])),
        selected_layer_forwards=d['query_selected_layers'],scanned_cold_rows=d['query_scanned_rows'],
        control_native_h2d_bytes=old['prototype_counts']['host_read_bytes'],
        candidate_native_h2d_bytes=d['prototype_counts']['host_read_bytes'],
        candidate_native_d2h_bytes=d['prototype_counts']['host_write_bytes'],
        native_h2d_calls=d['native_h2d_calls'],native_d2h_calls=d['native_d2h_calls'],
        roots_hooks_released=True,all_owned_jobs_finished=True,full512mib_passed=active<=512,
        ready_for_development_gate=passing,ordinary_adoption=False,full_goal_complete=False,
        next='Frozen12development query-vsLRU pairs; nonnegative task-balanced direction required.' if passing else 'Stop expansion.',
        limits='Single model/seed fixed16K50 teacher audit. Equal teacher predictions do not prove free-generation quality; '
            'large full-scan native traffic is charged. No quiet timing, actual baseline equality, physicalSTT or ordinary adoption proof.',
        evidence_sha256={f:sha(BASE/f) for f in ('resource_plan.json','resource_implementation.json','resource_source.tar.gz',
            'resource.json','resource_cleanup.json','resource_job.json','verified_correctness.json','verify_resource.py')},
        frozen_control_sha256=sha(ROOT/plan['frozen_ordinary_control']))
    output=BASE/'verified_resource.json';assert not output.exists()
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
