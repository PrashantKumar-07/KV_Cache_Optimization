"""Genuine cold retention inside the same measured rounded host allocation."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    output=BASE/'verified_resource.json';assert not output.exists()
    p=read(BASE/'plan.json');d=read(BASE/'resource.json');old=read(ROOT/p['control'])
    assert sha(ROOT/p['control'])==p['control_sha256'] and sha(BASE/'source.tar.gz')==p['archive_sha256']
    with tarfile.open(BASE/'source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==p['sources_sha256'] and all(sha(ROOT/f)==v for f,v in sources.items())
    job=read(BASE/'job.json');assert job['status']=='finished' and job['exit_code']==0
    assert d['passed'] and d['layerwise_arm']=='control' and d['layerwise_hooks_removed']
    assert (d['hot'],d['cold'])==(991,1984) and d['prefill_last_logits']
    assert d['placement_rows']==d['placed_layers']==[]
    assert read(BASE/'resource_cleanup.json')==dict(root_hooks_restored=True,leases_closed=True)
    assert d['owned_pinned_after_cleanup_bytes']==d['owned_host_allocator_block_after_cleanup_bytes']==0
    for field in ('input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids'):
        assert d[field]==old[field],field
    assert len(d['predicted_token_ids'])==50 and len(old['predicted_token_ids'])==50
    for key in ('source_sha256','actual_source_sha256'):
        assert all(sources[f]==v for f,v in d[key].items())
    assert d['resource_model']['constant_model_storage_bytes']==old['resource_model']['constant_model_storage_bytes']
    rounded=d['owned_host_allocator_block_peak_bytes']
    requested=d['reserved_host_bytes']
    assert requested==d['owned_pinned_peak_bytes']==256*2**20
    same=rounded==old['owned_host_allocator_block_peak_bytes']==p['measured_host_capacity_bytes']
    ratio=d['occupied_host_bytes']/old['occupied_host_bytes']
    baseline=d['resource_model']['constant_model_storage_bytes']
    active=(d['gpu_phase_peaks']['decode_peak']-baseline+rounded)/2**20
    old_active=(old['gpu_phase_peaks']['decode_peak']-baseline+old['owned_host_allocator_block_peak_bytes'])/2**20
    passed=same and ratio>=1.5 and active<=576
    result=dict(verified=True,resource_pilot_passed=passed,
        control_cold_capacity=1090,candidate_cold_capacity=1984,
        control_requested_host_mib=old['reserved_host_bytes']/2**20,
        candidate_requested_host_mib=requested/2**20,
        control_actual_rounded_host_mib=old['owned_host_allocator_block_peak_bytes']/2**20,
        candidate_actual_rounded_host_mib=rounded/2**20,
        control_occupied_host_mib=old['occupied_host_bytes']/2**20,
        candidate_occupied_host_mib=d['occupied_host_bytes']/2**20,
        occupied_payload_ratio=ratio,control_active_decode_envelope_mib=old_active,
        candidate_active_decode_envelope_mib=active,shared_declared_active_ceiling_mib=576,
        candidate_reserved_envelope_mib=(d['gpu_phase_peaks']['whole_reserved_peak']-baseline+rounded)/2**20,
        control_prefill_peak_gb=old['gpu_phase_peaks']['prefill_peak']/1e9,
        candidate_prefill_peak_gb=d['gpu_phase_peaks']['prefill_peak']/1e9,
        changed_teacher_predictions=sum(a!=b for a,b in zip(d['predicted_token_ids'],old['predicted_token_ids'])),
        control_host_read_payload_bytes=old['prototype_counts']['host_read_bytes'],
        candidate_host_read_payload_bytes=d['prototype_counts']['host_read_bytes'],
        control_host_write_payload_bytes=old['prototype_counts']['host_write_bytes'],
        candidate_host_write_payload_bytes=d['prototype_counts']['host_write_bytes'],
        roots_hooks_released=True,all_owned_jobs_finished=True,ordinary_source_changed=False,
        ready_for_preregistered_development_quality=passed,ordinary_integration_ready=False,
        full512mib_passed=active<=512,exact_allocation_and_width_match=False,full_goal_complete=False,
        next='Run the preregistered12development quality pairs before any timing or adoption.' if passed else 'Stop expansion; resource pilot failed.',
        limits='Configuration changes retention/selection/predictions deliberately; this is not preservation against1090cold. '
            'Same rounded host range with more usable slots, not equal full resource use. GPU workspace/metadata and transfers charged. '
            'No current quality/latency win, independent-holdout tuning, physicalSTT or full P0 completion claim.',
        evidence_sha256={f:sha(BASE/f) for f in ('plan.json','source.tar.gz','resource.json','resource_cleanup.json','job.json')},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))


if __name__=='__main__':main()
