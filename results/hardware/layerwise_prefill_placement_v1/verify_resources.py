"""Frozen whole-prefill resource primary and exact teacher-stream guards."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
REPLAY=('predicted_token_ids','teacher_token_ids','input_ids_sha256','prompt_sha256',
        'teacher_source_sha256','state_sha256','counters','prototype_counts',
        'occupied_host_bytes','reserved_host_bytes','owned_pinned_peak_bytes',
        'owned_host_allocator_block_peak_bytes')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',choices=('shared','default'),required=True)
    geometry=parser.parse_args().geometry;output=BASE/f'verified_resources_{geometry}.json'
    assert not output.exists()
    plan=read(BASE/'plan.json');metadata=read(BASE/'implementation.json')
    assert sha(BASE/'plan.json')==metadata['plan_sha256']
    assert sha(BASE/'implementation_source.tar.gz')==metadata['archive_sha256']
    assert sha(BASE/'cpu.json')==metadata['cpu_evidence_sha256']
    with tarfile.open(BASE/'implementation_source.tar.gz') as archive:
        sources={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
    assert sources==metadata['sources']
    assert all(sha(ROOT/p)==v for p,v in sources.items())
    assert all(sources[p]==v for p,v in plan['control_source_sha256'].items())
    assert read(BASE/'cpu.json')['passed'] and len(read(BASE/'cpu.json')['pairs'])==8
    assert sha(ROOT/'results/hardware/prefill_allocation_audit_v1/verified_report.json')==plan['prerequisite_sha256']
    frozen=ROOT/f'results/hardware/prefill_last_logits_integration_v1/last_{geometry}.json'
    assert sha(frozen)==plan['frozen_resource_controls'][str(frozen.relative_to(ROOT))]
    old=read(frozen);control=read(BASE/f'control_{geometry}.json');candidate=read(BASE/f'candidate_{geometry}.json')
    jobs=read(BASE/'jobs.json')
    for arm,data in (('control',control),('candidate',candidate)):
        assert jobs[f'{arm}_{geometry}']['status']=='finished' and jobs[f'{arm}_{geometry}']['exit_code']==0
        assert data['passed'] and data['prefill_last_logits'] and data['layerwise_hooks_removed']
        assert data['layerwise_arm']==arm
        assert data['layerwise_driver_sha256']==sources[str((BASE/'audit_resources.py').relative_to(ROOT))]
        assert read(BASE/f'{arm}_{geometry}_cleanup.json')==dict(root_hooks_restored=True,leases_closed=True)
        assert data['owned_pinned_after_cleanup_bytes']==data['owned_host_allocator_block_after_cleanup_bytes']==0
        for field in REPLAY:assert data[field]==old[field],(arm,field)
        for key in ('source_sha256','actual_source_sha256'):
            assert all(sources[p]==v for p,v in data[key].items())
        assert len(data['predicted_token_ids'])==50
    assert control['gpu_phase_peaks']==old['gpu_phase_peaks']
    assert control['resource_model']['constant_model_storage_bytes']==candidate['resource_model']['constant_model_storage_bytes']
    assert [r['stage'] for r in control['boundaries']]==[r['stage'] for r in candidate['boundaries']]
    assert len(candidate['placement_rows'])==32 and candidate['placed_layers']==list(range(32))
    assert candidate['query_map_empty']
    assert all(row['layer']==i and row['raw_shape']==[1,8,16384,128] for i,row in enumerate(candidate['placement_rows']))
    a=control['gpu_phase_peaks']['prefill_peak'];b=candidate['gpu_phase_peaks']['prefill_peak']
    saving=1-b/a;passed=saving>=plan['primary']['minimum_relative_reduction']
    model_bytes=control['resource_model']['constant_model_storage_bytes']
    active=lambda d:(d['gpu_phase_peaks']['decode_peak']-model_bytes)/2**20
    result=dict(verified=True,geometry=geometry,memory_primary_passed=passed,
        control_prefill_peak_bytes=a,candidate_prefill_peak_bytes=b,relative_prefill_reduction=saving,
        control_active_decode_mib=active(control),candidate_active_decode_mib=active(candidate),
        control_reserved_whole_peak_bytes=control['gpu_phase_peaks']['whole_reserved_peak'],
        candidate_reserved_whole_peak_bytes=candidate['gpu_phase_peaks']['whole_reserved_peak'],
        prediction_state_counter_transfer_replay=True,all_jobs_finished=True,ordinary_source_changed=False,
        ready_for_next_untimed_gate=passed,ordinary_integration_ready=False,full_goal_complete=False,
        limits='One model/seed/full16K fixed teacher stream; frozen ordinary replay and unchanged measurement/reset scope. '
            'Generation/timing/lifetime GPU stress gates remain. No full512MiB, quality superiority, speed or physicalSTT claim.',
        evidence_sha256={p:sha(BASE/p) for p in ('plan.json','implementation.json','implementation_source.tar.gz','cpu.json','jobs.json',
            f'control_{geometry}.json',f'candidate_{geometry}.json',f'control_{geometry}_cleanup.json',f'candidate_{geometry}_cleanup.json')},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256',)},indent=2))


if __name__=='__main__':main()
