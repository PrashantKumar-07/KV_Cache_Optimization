"""Read-only audit of failed forward ownership primary and4K starting state."""
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from transformers.cache_utils import DynamicLayer

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((BASE/n).read_text())


def archive(name):
    with tarfile.open(BASE/name) as bundle:
        hashes={m.name:hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
                for m in bundle.getmembers() if m.isfile()}
    assert all(sha(ROOT/p)==value for p,value in hashes.items())
    return hashes


def main():
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    plan,phase_plan=read('plan.json'),read('phase_plan.json')
    assert sha(BASE/'source.tar.gz')==plan['source_bundle_sha256']
    assert sha(BASE/'phase_source.tar.gz')==phase_plan['source_bundle_sha256']
    assert sha(BASE/'shared.json')==phase_plan['pilot_sha256']
    sources,phase_sources=archive('source.tar.gz'),archive('phase_source.tar.gz')
    assert sources['experiments/raw_input_reuse.py']==plan['raw_policy_sha256']
    assert sources['experiments/forward_owner_release.py']==phase_sources['experiments/forward_owner_release.py']==phase_plan['owner_helper_sha256']
    assert hashlib.sha256(inspect.getsource(DynamicLayer.update).encode()).hexdigest()==plan['dynamic_layer_update_sha256']
    previous_dir=ROOT/'results/hardware/raw_input_reuse_screen_v1'
    previous_proof=json.loads((previous_dir/'verified_report.json').read_text())
    assert sha(previous_dir/'verified_report.json')==plan['control_proof_sha256']
    assert sha(previous_dir/'shared.json')==previous_proof['evidence_sha256']['shared.json']
    previous={(r['context'],r['arm']):r for r in json.loads((previous_dir/'shared.json').read_text())['rows']}
    jobs=read('jobs.json')
    assert len(jobs)==2 and all(j['status']=='finished' and j['exit_code']==0 for j in jobs)
    fields=('state_sha256','history_sha256','every_visible_state_sha256',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes',
        'occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls')
    trial_fields=('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps')
    pilot,phase=read('shared.json'),read('phases.json')
    measurements=[]
    for raw,hashes in ((pilot,sources),(phase,phase_sources)):
        assert raw['passed'] and raw['forward_owner_release']
        assert raw['actual_control']=='research_raw_input_reuse' and raw['actual_candidate']=='forward_owner_release'
        assert raw['owner_helper_sha256']==hashes['experiments/forward_owner_release.py']
        assert all(hashes[p]==value for p,value in raw['source_sha256'].items())
        assert (raw['config']['hot'],raw['config']['cold'],raw['config']['decode'])==(991,1090,12)
        for index,context in enumerate(raw['config']['contexts']):
            control,candidate=raw['rows'][index*2:index*2+2]
            assert control['context']==candidate['context']==context
            assert control['arm']=='fused_bounded' and candidate['arm']=='native_reuse'
            assert len(candidate['every_visible_state_sha256'])==13
            assert candidate['forward_owner_counts']==dict(released_forwards=12,restored_forwards=12,released_layers=384)
            for key in fields:assert control[key]==candidate[key],(context,key)
            for row in (control,candidate):
                assert len(row['trials'])==1 and len(row['trials'][0]['predicted_token_ids'])==12
                assert row['reserved_host_bytes']==151257088
                old=previous[(context,'native_reuse')]
                for key in fields:assert row[key]==old[key],('frozen replay',context,key)
                for key in trial_fields:
                    assert control['trials'][0][key]==candidate['trials'][0][key]
                    assert row['trials'][0][key]==old['trials'][0][key]
                resources=row['whole_decode_resources']
                assert resources['root']['allocator_block_bytes']==256*2**20
                for kind in ('allocated','reserved'):
                    assert resources[f'{kind}_extra_plus_host_range_bytes']==resources[f'gpu_{kind}_peak_bytes']-resources['constant_model_storage_bytes']+256*2**20
            if raw is pilot:
                before=control['whole_decode_resources']['allocated_extra_plus_host_range_bytes']
                after=candidate['whole_decode_resources']['allocated_extra_plus_host_range_bytes']
                gain=(before-after)/before*100
                assert gain<10 and after>512*2**20
                measurements.append(dict(context=context,control_active_envelope_mib=before/2**20,
                    candidate_active_envelope_mib=after/2**20,reduction_pct=gain,primary_passed=False))
    assert pilot['config']['contexts']==[4096,8192,16384,31500] and len(pilot['rows'])==8
    assert phase['config']['contexts']==[4096] and len(phase['rows'])==2
    assert phase['owner_phase_sha256']==phase_sources['experiments/audit_forward_owner_phases.py']
    frames=phase['owner_phase_frames'];assert len(frames)==52
    peak=phase['rows'][1]['whole_decode_resources']['gpu_allocated_peak_bytes']
    decoder=frames[4:];assert len(decoder)==48
    before,after,forward,restored=decoder[:4]
    assert after['peak_allocated_bytes']==peak and before['peak_allocated_bytes']<peak
    release=(before['allocated_bytes']-after['allocated_bytes'])/2**20
    assert release>127
    assert after['allocated_bytes']<before['allocated_bytes']
    assert peak-before['allocated_bytes']<=2*2**20
    for step in range(12):
        before,after,forward,restored=decoder[step*4:step*4+4]
        assert [r['stage'] for r in (before,after,forward,restored)]==[
            'lease_before','lease_after','forward_before_restore','forward_after_restore']
        assert all(r['input_rows']==1 for r in (before,after,forward,restored))
        assert before['released_forwards']==before['restored_forwards']==step
        assert after['released_forwards']==forward['released_forwards']==restored['released_forwards']==step+1
        assert restored['restored_forwards']==step+1 and not restored['pending']
        assert after['pending'] and forward['pending']
        assert forward['peak_allocated_bytes']==restored['peak_allocated_bytes']==peak
        assert forward['allocated_bytes']==restored['allocated_bytes']
    assert '227 passed' in (BASE/'full_suite.txt').read_text()
    assert '10 passed' in (BASE/'focused_tests.txt').read_text()
    proof=dict(verified=True,decision='reject',primary_passed=False,contexts=measurements,
        exact_pairs=4,frozen_replay_rows=8,full_suite_passed=227,focused_exception_owner_tests=10,
        released_and_restored_layers=1536,phase_context=4096,
        first_release_live_gpu_reduction_mib=release,
        peak_reached_in_first_pre_forward_validation=True,
        advance_to_usual_budget=False,advance_to_timing_or_generation=False,
        ordinary_candidate_ready=False,complete_resource_equality=False,
        evidence_sha256={n:sha(BASE/n) for n in ('plan.json','source.tar.gz','shared.json','phase_plan.json','phase_source.tar.gz','phases.json','jobs.json','full_suite.txt','focused_tests.txt')},
        verifier_sha256=sha(Path(__file__)),
        limitations='One model/seed/reused teacher/shared capacities; four-context primary fails.4K-only phase attribution. Owner release tested for partial-forward CPU exceptions/padded views/fallback; macro generation/lifetime/serving stress not advanced. Torch+owned-host scope excludes other host/cache/driver/non-Torch allocations. Timings invalid, no ordinary/default/quality/SnapKV/physicalSTT/completion claim.')
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(verified=True,decision='reject',contexts=measurements,full_suite_passed=227),indent=2))


if __name__=='__main__':main()
