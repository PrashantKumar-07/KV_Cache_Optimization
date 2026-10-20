"""Read-only prefill ownership memory gate; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((BASE/n).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shared-only',action='store_true');args=parser.parse_args()
    output=BASE/('verified_shared.json' if args.shared_only else 'verified_pilot.json')
    if output.exists():raise FileExistsError(output)
    plan=read('plan.json');assert sha(BASE/'source.tar.gz')==plan['source_bundle_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        sources={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
    assert all(sha(ROOT/p)==value for p,value in sources.items())
    assert sources['experiments/forward_owner_release.py']==plan['owner_helper_sha256']
    assert sources['experiments/raw_input_reuse.py']==plan['raw_policy_sha256']
    previous_dir=ROOT/'results/hardware/forward_owner_release_v1'
    previous_proof=json.loads((previous_dir/'verified_report.json').read_text())
    assert sha(previous_dir/'verified_report.json')==plan['control_proof_sha256']
    assert sha(previous_dir/'shared.json')==previous_proof['evidence_sha256']['shared.json']
    fields=('state_sha256','history_sha256','every_visible_state_sha256',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes',
        'occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls')
    trial_fields=('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps')
    geometries=[('shared',991,1090,256)] if args.shared_only else [('shared',991,1090,256),('default',1024,2048,512)]
    summary={};replayed=0
    for name,hot,cold,host_mib in geometries:
        raw=read(name+'.json')
        assert raw['passed'] and raw['prefill_owner_share'] and len(raw['rows'])==8
        assert raw['actual_control']=='research_forward_owner_release' and raw['actual_candidate']=='prefill_owner_share'
        assert raw['prefill_helper_sha256']==sources['experiments/prefill_owner_share.py']
        assert all(sources[p]==value for p,value in raw['source_sha256'].items())
        assert (raw['config']['hot'],raw['config']['cold'],raw['config']['decode'])==(hot,cold,12)
        assert raw['config']['contexts']==[4096,8192,16384,31500] and raw['config']['warmup']==raw['config']['blocks']==0
        if name=='shared':
            prior_path=previous_dir/'shared.json';old_arm='native_reuse'
        else:
            prior_dir=ROOT/'results/hardware/promotion_planning_screen_20261002'
            prior_path=prior_dir/'model_parity.json'
            proof=json.loads((prior_dir/'verified_model_parity.json').read_text())
            assert sha(prior_path)==proof['evidence_sha256'][prior_path.name]
            old_arm='native_reuse'
        old={(r['context'],r['arm']):r for r in json.loads(prior_path.read_text())['rows']}
        measurements=[]
        for i,context in enumerate(raw['config']['contexts']):
            control,candidate=raw['rows'][i*2:i*2+2]
            assert control['context']==candidate['context']==context
            assert control['arm']=='fused_bounded' and candidate['arm']=='native_reuse'
            assert candidate['prefill_shared_layers']==32
            for key in fields:assert control[key]==candidate[key],(name,context,key)
            for row in (control,candidate):
                assert row['owner_hooks_removed'] and row['forward_owner_counts']==dict(released_forwards=12,restored_forwards=12,released_layers=384)
                assert len(row['every_visible_state_sha256'])==13 and len(row['trials'])==1
                assert row['reserved_host_bytes']==(cold+64)*131072
                previous=old[(context,old_arm)]
                for key in fields:assert row[key]==previous[key],('frozen replay',name,context,key)
                for key in trial_fields:
                    assert control['trials'][0][key]==candidate['trials'][0][key]
                    assert row['trials'][0][key]==previous['trials'][0][key]
                resources=row['whole_decode_resources']
                assert resources['root']['allocator_block_bytes']==host_mib*2**20
                for kind in ('allocated','reserved'):
                    assert resources[f'{kind}_extra_plus_host_range_bytes']==resources[f'gpu_{kind}_peak_bytes']-resources['constant_model_storage_bytes']+host_mib*2**20
                replayed+=1
            before=control['whole_decode_resources']['allocated_extra_plus_host_range_bytes']
            after=candidate['whole_decode_resources']['allocated_extra_plus_host_range_bytes']
            gain=(before-after)/before*100
            assert gain>=10
            if name=='shared':assert after<512*2**20
            else:assert after>512*2**20
            measurements.append(dict(context=context,control_active_envelope_mib=before/2**20,
                candidate_active_envelope_mib=after/2**20,reduction_pct=gain,
                candidate_reserved_envelope_mib=candidate['whole_decode_resources']['reserved_extra_plus_host_range_bytes']/2**20,
                prefill_peak_gb=candidate['prefill_peak_memory_gb']))
        summary[name]=measurements
    assert '236 passed' in (BASE/'full_suite.txt').read_text()
    assert '9 passed' in (BASE/'focused_tests.txt').read_text()
    names=[name+'.json' for name,_,_,_ in geometries]+['plan.json','source.tar.gz','full_suite.txt','focused_tests.txt']
    proof=dict(verified=True,primary_passed=True,geometries=summary,frozen_replay_rows=replayed,
        full_suite_passed=236,focused_prefill_tests=9,advance_to_usual_budget=args.shared_only,
        advance_to_timing=not args.shared_only,ordinary_candidate_ready=False,complete_resource_equality=False,
        evidence_sha256={n:sha(BASE/n) for n in names},verifier_sha256=sha(Path(__file__)),
        limitations='Direction-only fixed teacher pilot/one model/seed. Both arms use research forward leases and standardTorch payload compaction. Allocated-minus-constant-model-plus-owned-host observed envelope excludes other host/cache/driver/non-Torch scopes; retained reservation/whole-request/prefill remain separate. Shared observed active fits512; usual host512 alone leaves no GPU budget. No timing/generation/serving/quality/physicalSTT/ordinary/default/completion claim.')
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(verified=True,primary_passed=True,geometries=summary,full_suite_passed=236),indent=2))


if __name__=='__main__':main()
