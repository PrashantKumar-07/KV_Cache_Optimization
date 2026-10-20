"""Read-only audit of frozen bulk-promotion integrity results."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda name:json.loads((BASE/name).read_text())


def main():
    output=BASE/'verified_integrity.json'
    if output.exists():raise FileExistsError(output)
    plan=read('integrity_plan.json')
    assert sha(BASE/'integrity_source.tar.gz')==plan['source_bundle_sha256']
    assert sha(BASE/'verified_model_timing.json')==plan['timing_proof_sha256']
    jobs=read('integrity_jobs.json')
    assert jobs['passed'] and len(jobs['jobs'])==7
    assert all(j['status']=='finished' and j['exit_code']==0 for j in jobs['jobs'])
    with tarfile.open(BASE/'integrity_source.tar.gz') as archive:
        sources={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
    names=['native']+[f'{kind}_{g}' for kind in ('boundary','generation','interleaved') for g in ('shared','default')]
    for name in names:
        raw=read(name+'.json')
        assert raw['passed'] and raw['bulk_promotion'] and raw['actual_candidate']=='bulk_promotion_native_reuse'
        for key in ('source_sha256','reuse_source_sha256','bulk_integrity_source_sha256','fused_source_sha256'):
            assert all(sources[p]==value for p,value in raw.get(key,{}).items()),(name,key)
        assert raw['reuse_binary_sha256']==read('model_parity.json')['reuse_binary_sha256']
    native=read('native.json')
    assert len(native['native_checks'])==15 and all(c['exact'] for c in native['native_checks'])
    fields=('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
        'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes')
    counts={};replayed=0
    for geometry,hot,cold in (('shared',991,1090),('default',1024,2048)):
        raw=read(f'generation_{geometry}.json')
        manifest_path=Path(plan['generation_manifests'][geometry]['path'])
        assert sha(manifest_path)==plan['generation_manifests'][geometry]['sha256']==raw['manifest_sha256']
        manifest=json.loads(manifest_path.read_text())
        expected=[(task,sample) for task,entry in manifest['tasks'].items() for sample in entry['selected_sample_ids']]
        assert len(expected)==12 and len(raw['rows'])==24 and raw['hot']==hot and raw['cold']==cold
        assert raw['actual_control']=='ordinary_native_reuse'
        old=json.loads(Path(f'results/hardware/reuse_visible_integrity_20261002/generation_{geometry}.json').read_text())
        previous={(r['task'],r['sample_id']):r for r in old['rows'] if r['arm']=='native_reuse'}
        for index,(task,sample) in enumerate(expected):
            pair=raw['rows'][2*index:2*index+2]
            assert {r['arm'] for r in pair}=={'fused_bounded','native_reuse'}
            assert all(r['task']==task and r['sample_id']==sample and r['seed']==0 for r in pair)
            for field in fields:assert pair[0][field]==pair[1][field],(geometry,sample,field)
            for row in pair:
                assert row['reserved_host_tensor_bytes']==(cold+64)*131072
                assert row['termination'] in ('eos','cap') and len(row['generated_token_ids'])<=row['max_gen']
                for field in fields:assert row[field]==previous[(task,sample)][field],('prior frozen replay',geometry,sample,field)
                replayed+=1
            candidate=next(r for r in pair if r['arm']=='native_reuse')
            assert candidate['reuse_counts']['reused_layers']==candidate['reuse_counts']['reused_compactions']
        boundary=read(f'boundary_{geometry}.json')
        assert len(boundary['checks'])==(36 if geometry=='shared' else 4)
        assert boundary['full_transient_saturation_observed'] and boundary['native_reuse_launches']['calls']>0
        assert max(r['transient_logical_peak'] for r in boundary['checks'])==cold+64
        assert all(r['exact_state_payload_counters'] and r['rejected_append_before_mutation'] and r['decode_calls']==24 for r in boundary['checks'])
        assert sum(r['eager_compactions'] for r in boundary['checks'])>0
        inter=read(f'interleaved_{geometry}.json')
        assert inter['hot']==hot and inter['cold']==cold and len(inter['rows'])==9
        assert inter['native_reuse_launches']=={'calls':48,'layers':1536}
        for request,frames in (('A',13),('B',25),('C',13)):
            values=[r for r in inter['rows'] if r['request']==request]
            assert len(values)==3 and {r['mode'] for r in values}=={'gpu','cpu','interleaved_cpu'}
            assert all(len(r['every_boundary_sha256'])==frames for r in values)
            for field in ('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters'):
                assert all(r[field]==values[0][field] for r in values),('request',geometry,request,field)
        counts[geometry]=dict(generation_pairs=12,boundary_cases=len(boundary['checks']),recycled_requests=3,boundary_frames_per_mode=51)
    assert '199 passed' in (BASE/'full_suite.txt').read_text()
    proof=dict(verified=True,native_bit_stream_checks=15,geometries=counts,frozen_generation_replay_rows=replayed,
        full_suite_passed=199,candidate_ready_for_cli=False,advance_to_cli_after_shared_timing_guards=True,
        evidence_sha256={name+'.json':sha(BASE/(name+'.json')) for name in names}|{
            n:sha(BASE/n) for n in ('integrity_plan.json','integrity_source.tar.gz','integrity_jobs.json','full_suite.txt')},
        verifier_sha256=sha(Path(__file__)),limitations='One model/seed/reused development set; exact preservation only. Native kernels unchanged. Serialized interleaving is not threaded serving or throughput. Shared timing and ordinary CLI adoption still required. No quality/physical STT/SnapKV/actual-resource superiority claim.')
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(verified=True,frozen_generation_replay_rows=replayed,geometries=counts),indent=2))


if __name__=='__main__':main()
