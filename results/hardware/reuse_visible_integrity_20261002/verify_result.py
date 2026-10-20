"""Independently check native reuse preservation, sources and request isolation."""
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((BASE/'plan.json').read_text())
    request_plan=json.loads((BASE/'request_plan.json').read_text())
    assert sha(BASE/'source.tar.gz')==plan['source_archive_sha256']==request_plan['source_archive_sha256']
    assert sha(Path('results/hardware/reuse_visible_model_20261002/verified_timing.json'))==plan['timing_proof_sha256']
    manifest={}
    for name in ('source.tar.gz','request_source.tar.gz'):
        with tarfile.open(BASE/name) as archive:
            for member in archive.getmembers():
                if member.isfile():manifest[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert manifest['experiments/check_reuse_requests.py']==request_plan['driver_sha256']
    paths=['native.json']+[f'{kind}_{geometry}.json' for kind in ('generation','boundary','interleaved') for geometry in ('shared','default')]
    for name in paths:
        d=json.loads((BASE/name).read_text());assert d['passed'] and not d.get('error'),name
        assert d['reuse_binary_sha256']==plan['native_binary_sha256']
        sources=d.get('reuse_source_sha256',d['source_sha256'])
        assert all(manifest[p]==value for p,value in sources.items()),name
    native=json.loads((BASE/'native.json').read_text())
    assert len(native['native_checks'])==15 and all(r['exact'] for r in native['native_checks'])
    delayed=[r for r in native['native_checks'] if r['case']=='delayed_source_destination_release_allocator_pressure']
    assert len(delayed)==12 and [r['consumer'] for r in delayed]==[0,1]*6
    rows_total=0
    counts={}
    equality_fields=('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
        'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes')
    for geometry,hot,cold in (('shared',991,1090),('default',1024,2048)):
        raw=json.loads((BASE/f'generation_{geometry}.json').read_text())
        mp=Path(f'results/hardware/fused_bounded_pool_20261002/generation_{geometry}/generation_manifest.json')
        m=json.loads(mp.read_text());assert sha(mp)==raw['manifest_sha256']
        assert raw['hot']==hot and raw['cold']==cold and m['seeds']==[0]
        expected=[(task,sample) for task,entry in m['tasks'].items() for sample in entry['selected_sample_ids']]
        assert len(raw['rows'])==24 and len(expected)==12
        for index,(task,sample) in enumerate(expected):
            pair=raw['rows'][2*index:2*index+2]
            assert {r['arm'] for r in pair}=={'fused_bounded','native_reuse'}
            assert all(r['task']==task and r['sample_id']==sample and r['seed']==0 for r in pair)
            for field in equality_fields:assert pair[0][field]==pair[1][field],(geometry,task,field)
            for row in pair:
                assert row['reserved_host_tensor_bytes']==(cold+64)*131072
                assert row['termination'] in ('eos','cap')
                assert len(row['generated_token_ids'])<=row['max_gen']
            candidate=next(r for r in pair if r['arm']=='native_reuse')
            assert candidate['reuse_counts']['reused_layers']==candidate['reuse_counts']['reused_compactions']
        rows_total+=len(raw['rows'])
        boundary=json.loads((BASE/f'boundary_{geometry}.json').read_text())
        assert len(boundary['checks'])==(36 if geometry=='shared' else 4)
        assert boundary['full_transient_saturation_observed'] and boundary['native_reuse']
        assert boundary['native_reuse_launches']['calls']>0
        assert max(r['transient_logical_peak'] for r in boundary['checks'])==cold+64
        assert all(r['exact_state_payload_counters'] and r['rejected_append_before_mutation'] and r['decode_calls']==24 for r in boundary['checks'])
        assert sum(r['eager_compactions'] for r in boundary['checks'])>0
        inter=json.loads((BASE/f'interleaved_{geometry}.json').read_text())
        assert inter['hot']==hot and inter['cold']==cold and inter['native_reuse']
        assert len(inter['rows'])==9 and inter['native_reuse_launches']=={'calls':48,'layers':1536}
        for request,frames in (('A',13),('B',25),('C',13)):
            values=[r for r in inter['rows'] if r['request']==request]
            assert len(values)==3 and {r['mode'] for r in values}=={'gpu','cpu','interleaved_cpu'}
            for field in ('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters'):
                assert values[0][field]==values[1][field]==values[2][field],(geometry,request,field)
            assert all(len(r['every_boundary_sha256'])==frames for r in values)
        counts[geometry]={'generation_pairs':12,'boundary_cases':len(boundary['checks']),'interleaved_frames_per_mode':51}
    report=dict(verified=True,native_bit_and_stream_checks=15,generation_rows=rows_total,geometries=counts,
        reuse_binary_sha256=plan['native_binary_sha256'],native_source_sha256={p:manifest[p] for p in ('experiments/native_hot_compact.cpp','experiments/native_reuse_hot.cu')},torch_version=native['runtime']['packages']['torch'],python_major_minor=[3,10],candidate_ready_for_opt_in_cli=True,default_change=False,new_speed_claim=False,whole_model_memory_claim=False,
        evidence_sha256={p:sha(BASE/p) for p in paths+['plan.json','request_plan.json','source.tar.gz','request_source.tar.gz']},
        verifier_sha256=sha(Path(__file__)),limitations='One model/seed/reused development set. Known policy KV lower bound and unchanged whole-request peak. Explicit-event native stream tests and serialized request recycling do not prove threaded serving or throughput. Retention and SnapKV superiority gates remain failed; CLI gate still pending.')
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
