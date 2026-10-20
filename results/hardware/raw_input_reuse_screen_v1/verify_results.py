"""Read-only audit of a rejected payload prototype and4K phase diagnosis."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n:json.loads((BASE/n).read_text())


def archive(name):
    with tarfile.open(BASE/name) as bundle:
        hashes = {m.name:hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
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
    assert sources['experiments/raw_input_reuse.py']==phase_sources['experiments/raw_input_reuse.py']==phase_plan['raw_policy_sha256']
    prefix=ROOT/'results/hardware/decode_prefix_audit_v1/verified_report.json'
    cli=ROOT/'results/hardware/promotion_planning_screen_20261002/cli/verified_report.json'
    assert sha(prefix)==plan['prefix_proof_sha256'] and sha(cli)==plan['control_cli_proof_sha256']
    jobs=read('jobs.json')
    assert len(jobs)==2 and all(j['status']=='finished' and j['exit_code']==0 for j in jobs)
    fields=('state_sha256','history_sha256','every_visible_state_sha256',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes',
        'occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls')
    trial_fields=('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps')
    pilot,phase=read('shared.json'),read('phases.json')
    for raw,hashes in ((pilot,sources),(phase,phase_sources)):
        assert raw['passed'] and raw['raw_input_reuse']
        assert raw['actual_control']=='ordinary_bulk_promotion' and raw['actual_candidate']=='raw_input_reuse'
        assert raw['raw_policy_sha256']==hashes['experiments/raw_input_reuse.py']
        assert all(hashes[p]==value for p,value in raw['source_sha256'].items())
        assert (raw['config']['hot'],raw['config']['cold'],raw['config']['decode'])==(991,1090,12)
        for i,context in enumerate(raw['config']['contexts']):
            control,candidate=raw['rows'][i*2:i*2+2]
            assert control['context']==candidate['context']==context
            assert control['arm']=='fused_bounded' and candidate['arm']=='native_reuse'
            assert len(candidate['every_visible_state_sha256'])==13
            for key in fields:assert control[key]==candidate[key],(context,key)
            for row in (control,candidate):
                assert len(row['trials'])==1 and len(row['trials'][0]['predicted_token_ids'])==12
                assert row['reserved_host_bytes']==151257088
                for key in trial_fields:assert control['trials'][0][key]==candidate['trials'][0][key]
                w=row['whole_decode_resources']
                assert w['root']['allocator_block_bytes']==256*2**20
                for kind in ('allocated','reserved'):
                    assert w[f'{kind}_extra_plus_host_range_bytes']==w[f'gpu_{kind}_peak_bytes']-w['constant_model_storage_bytes']+w['root']['allocator_block_bytes']
    assert pilot['config']['contexts']==[4096,8192,16384,31500] and len(pilot['rows'])==8
    assert phase['config']['contexts']==[4096] and len(phase['rows'])==2
    prior_path=ROOT/'results/hardware/decode_prefix_audit_v1/shared.json'
    prefix_proof=json.loads(prefix.read_text())
    assert sha(prior_path)==prefix_proof['evidence_sha256']['shared.json']
    previous={(r['context'],r['arm']):r for r in json.loads(prior_path.read_text())['rows']}
    measurements=[]
    for index,context in enumerate(pilot['config']['contexts']):
        control,candidate=pilot['rows'][index*2:index*2+2]
        for row in (control,candidate):
            old=previous[(context,row['arm'])]
            for key in fields:assert row[key]==old[key],('frozen replay',context,key)
            for key in trial_fields:assert row['trials'][0][key]==old['trials'][0][key]
        before=control['whole_decode_resources']['allocated_extra_plus_host_range_bytes']
        after=candidate['whole_decode_resources']['allocated_extra_plus_host_range_bytes']
        gain=(before-after)/before*100
        assert gain<10 and after>512*2**20
        measurements.append(dict(context=context,control_active_envelope_mib=before/2**20,
            candidate_active_envelope_mib=after/2**20,reduction_pct=gain,primary_passed=False))
    assert phase['raw_phase_source_sha256']==phase_sources['experiments/audit_raw_reuse_phases.py']
    frames=phase['raw_phase_frames'];forwards=phase['raw_forward_frames']
    assert len(frames)==37 and len(forwards)==26
    assert [(r['call'],r['input_rows']) for r in forwards]==[(i,4096 if i in (1,14) else 1) for i in range(1,27)]
    candidate_forwards=forwards[14:]
    candidate_peak=phase['rows'][1]['whole_decode_resources']['gpu_allocated_peak_bytes']
    peak_step=next(i for i,row in enumerate(candidate_forwards,1) if row['peak_allocated_bytes']==candidate_peak)
    assert peak_step==2
    releases=[]
    for step in range(1,13):
        enter,exit,visible=[r for r in frames if r['step']==step]
        assert [r['stage'] for r in (enter,exit,visible)]==['prepare_entry','prepare_exit','visible_exit']
        assert enter['bound_layers']==enter['compactions']==(step-1)*32
        assert exit['bound_layers']==visible['bound_layers']==visible['compactions']==step*32
        assert all(r['peak_allocated_bytes']==candidate_forwards[step-1]['peak_allocated_bytes'] for r in (enter,exit,visible))
        assert enter['allocated_bytes']==candidate_forwards[step-1]['allocated_bytes']
        releases.append((enter['allocated_bytes']-exit['allocated_bytes'])/2**20)
    assert all(value>=127 for value in releases)
    for row,old in zip(phase['rows'],pilot['rows'][:2]):
        for key in fields:assert row[key]==old[key],('observer preservation',key)
        for key in trial_fields:assert row['trials'][0][key]==old['trials'][0][key]
    assert '217 passed' in (BASE/'full_suite.txt').read_text()
    proof=dict(verified=True,decision='reject',primary_passed=False,contexts=measurements,
        exact_pairs=4,frozen_replay_rows=8,full_suite_passed=217,focused_payload_tests=11,
        phase_context=4096,actual_bindings=384,actual_compactions=384,
        peak_first_reached_in_forward_step=peak_step,peak_reached_before_same_step_candidate_prepare=True,
        prepare_live_release_mib=releases,
        advance_to_usual_budget=False,advance_to_timing_or_generation=False,ordinary_candidate_ready=False,
        complete_resource_equality=False,
        evidence_sha256={n:sha(BASE/n) for n in ('plan.json','source.tar.gz','shared.json','phase_plan.json','phase_source.tar.gz','phases.json','jobs.json','full_suite.txt')},
        verifier_sha256=sha(Path(__file__)),
        limitations='One model/seed/reused teacher,shared capacities.4context primary fails;4K-only phase attribution. Public Torch+live owned host range excludes other host/cache/driver/non-Torch scopes. Recorded observer timings invalid. StandardTorch alias-safe per-layer scratch tested, but native/generation/concurrency stress and full serving support were not advanced. No ordinary/default/speed/quality/physicalSTT/completion claim.')
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(verified=True,decision='reject',contexts=measurements,actual_bindings=384,full_suite_passed=217),indent=2))


if __name__=='__main__':main()
