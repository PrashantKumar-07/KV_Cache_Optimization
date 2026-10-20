"""Resource and copied channel-payload ledger against the immutable suffix run."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    output=BASE/'verified_resource.json';assert not output.exists()
    plan=read(BASE/'resource_plan.json');metadata=read(BASE/'resource_implementation.json');raw=read(BASE/'resource.json')
    assert metadata['plan_sha256']==sha(BASE/'resource_plan.json')
    assert plan['correctness_proof_sha256']==sha(BASE/'verified_correctness.json')
    assert metadata['archive_sha256']==sha(BASE/'resource_source.tar.gz')
    with tarfile.open(BASE/'resource_source.tar.gz') as archive:
        sources={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
    assert sources==metadata['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    assert raw['passed'] and (raw['hot'],raw['cold'])==(991,1090) and raw['prefill_last_logits']
    assert read(BASE/'resource_job.json')['status']=='finished' and read(BASE/'resource_job.json')['exit_code']==0
    assert read(BASE/'resource_cleanup.json')==dict(selectors_closed=True,attention_pre_hooks_restored=True)
    assert raw['query_hooks_removed'] and raw['index_metadata_released']
    assert raw['owned_pinned_after_cleanup_bytes']==raw['owned_host_allocator_block_after_cleanup_bytes']==0
    old=read(ROOT/plan['reference']);assert sha(ROOT/plan['reference'])==plan['reference_sha256']
    for key in ['input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids',
                'predicted_token_ids','state_sha256','counters']:
        assert raw[key]==old[key],key
    for key in ['source_sha256','actual_source_sha256']:assert all(sources[f]==h for f,h in raw[key].items())
    assert len(raw['predicted_token_ids'])==50
    assert raw['query_selected_layers']==raw['visible_suffix_updates']==raw['channel_key_reads']==raw['channel_value_reads']==1600
    assert raw['index_reads']==3200 and raw['index_refreshes']==51
    assert raw['channel_key_rows']==raw['query_scanned_rows']==old['query_scanned_rows']
    assert raw['channel_value_rows']==1600*32
    rb=8*128*2;key_bytes=raw['channel_key_rows']*rb;value_bytes=raw['channel_value_rows']*rb
    common=old['prototype_counts']['host_read_bytes']-2*key_bytes
    assert common>=0 and raw['prototype_counts']['host_read_bytes']==common+key_bytes+value_bytes
    for key in ['host_write_bytes','host_initial_write_bytes']:assert raw['prototype_counts'][key]==old['prototype_counts'][key]
    assert raw['native_h2d_calls']==old['native_h2d_calls']+1600==3301
    assert raw['native_d2h_calls']==old['native_d2h_calls']==78
    assert raw['channel_native_binary_sha256']==read(BASE/'native_v2.json')['binary_sha256']
    assert sha(Path(read(BASE/'native_v2.json')['binary_path']))==raw['channel_native_binary_sha256']
    assert raw['index_gpu_storage_bytes']==old['index_gpu_storage_bytes']==1477120
    assert raw['index_metadata_h2d_bytes']==old['index_metadata_h2d_bytes']==75333120
    const=raw['resource_model']['constant_model_storage_bytes'];assert const==old['resource_model']['constant_model_storage_bytes']
    host=raw['owned_host_allocator_block_peak_bytes'];assert host==old['owned_host_allocator_block_peak_bytes']==256*2**20
    assert raw['reserved_host_bytes']==raw['owned_pinned_peak_bytes']==151257088
    active=(raw['gpu_phase_peaks']['decode_peak']-const+host)/2**20
    control=(old['gpu_phase_peaks']['decode_peak']-const+host)/2**20
    passing=active<=576
    result=dict(verified=True,resource_primary_passed=passing,declared_active_ceiling_mib=576,
        control_active_decode_envelope_mib=control,candidate_active_decode_envelope_mib=active,
        candidate_reserved_envelope_mib=(raw['gpu_phase_peaks']['whole_reserved_peak']-const+host)/2**20,
        candidate_prefill_peak_gb=raw['gpu_phase_peaks']['prefill_peak']/1e9,rounded_owned_host_mib=host/2**20,
        changed_teacher_predictions=0,selected_layer_forwards=1600,visible_suffix_updates=1600,
        full512mib_passed=active<=512,channel_key_rows=raw['channel_key_rows'],channel_value_rows=raw['channel_value_rows'],
        query_key_h2d_bytes=key_bytes,query_selected_value_h2d_bytes=value_bytes,common_nonquery_kv_h2d_bytes=common,
        control_copied_kv_h2d_bytes=old['prototype_counts']['host_read_bytes'],candidate_copied_kv_h2d_bytes=raw['prototype_counts']['host_read_bytes'],
        copied_kv_payload_reduction=1-raw['prototype_counts']['host_read_bytes']/old['prototype_counts']['host_read_bytes'],
        unchanged_native_d2h_bytes=raw['prototype_counts']['host_write_bytes'],native_h2d_calls=3301,native_d2h_calls=78,
        channel_key_reads=1600,channel_value_reads=1600,index_refreshes=51,
        unchanged_batch_index_metadata_h2d_bytes=75333120,native_binary_sha256=raw['channel_native_binary_sha256'],
        roots_hooks_and_index_metadata_released=True,all_owned_jobs_finished=True,
        ready_for_component_cost_gate=passing,development_quality_expansion=False,ordinary_adoption=False,full_goal_complete=False,
        next='Preregister3ABBA channel-vs-paired suffix component cost; ordinaryLRU guard required before wider retrieval/adoption.' if passing else 'Stop expansion.',
        limits=plan['limits'],evidence_sha256={name:sha(BASE/name) for name in ['resource_plan.json','resource_implementation.json',
            'resource_source.tar.gz','resource.json','resource_cleanup.json','resource_job.json','verified_correctness.json','native_v2.json','verify_resource.py']},
        frozen_control_sha256=plan['reference_sha256'])
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
