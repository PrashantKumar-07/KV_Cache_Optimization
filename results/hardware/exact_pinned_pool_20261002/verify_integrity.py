"""Independent preservation/lifetime verification; failed first hook run retained."""
import hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifests={}
    bundles=['source_model_v2.tar.gz','integrity_resource_source.tar.gz','integrity_retry_source.tar.gz']
    for name in bundles:
        with tarfile.open(B/name) as archive:
            for member in archive.getmembers():
                if member.isfile():manifests.setdefault(member.name,set()).add(hashlib.sha256(archive.extractfile(member).read()).hexdigest())
    plan=json.loads((B/'integrity_plan.json').read_text())
    retry=json.loads((B/'integrity_retry_plan.json').read_text())
    assert retry['driver_sha256'] in manifests['experiments/check_exact_pool_requests.py']
    assert all(v in manifests[p] for p,v in plan['driver_sha256'].items())
    failed=json.loads((B/'interleaved_shared.json').read_text())
    assert sha(B/'interleaved_shared.json')==retry['failed_raw_sha256']
    assert failed['native_reuse_launches']=={'calls':0,'layers':0}
    assert 'AssertionError' in (B/'interleaved_shared.log').read_text()
    evidence=[];geometries={}
    generation_fields=('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
        'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes')
    for name,hot,cold in [('shared',991,1090),('default',1024,2048)]:
        files=[f'generation_{name}.json',f'interleaved_{name}'+('_v2' if name=='shared' else '')+'.json']
        raw=[]
        for filename in files:
            evidence.append(filename);d=json.loads((B/filename).read_text());raw.append(d)
            assert d['passed'] and not d.get('error'),filename
            sources=d.get('exact_generation_source_sha256',d.get('exact_pool_source_sha256'))
            assert all(v in manifests[p] for p,v in sources.items()),filename
            s=d['exact_allocation_stats'];assert s['live_allocations']==s['live_requested_bytes']==0
            allocations={};releases=set()
            for event in s['events']:
                token=event['token'];assert event['bytes']==(cold+64)*131072
                if event['kind']=='allocate':
                    assert token not in allocations;allocations[token]=event['bytes']
                else:
                    assert event['kind']=='release' and token in allocations and token not in releases
                    releases.add(token)
            assert releases==set(allocations)
            assert len(allocations)==(12 if filename.startswith('generation') else 3)
        gen,inter=raw
        mp=Path(f'results/hardware/fused_bounded_pool_20261002/generation_{name}/generation_manifest.json')
        m=json.loads(mp.read_text());assert sha(mp)==gen['manifest_sha256']
        assert gen['hot']==hot and gen['cold']==cold and m['seeds']==[0]
        assert gen['arm_mapping']=={'fused_bounded':'cached_native_reuse','native_reuse':'exact_native_reuse'}
        expected=[(task,sample) for task,entry in m['tasks'].items() for sample in entry['selected_sample_ids']]
        assert len(expected)==12 and len(gen['rows'])==24
        for index,(task,sample) in enumerate(expected):
            pair=gen['rows'][index*2:index*2+2]
            assert {r['arm'] for r in pair}=={'fused_bounded','native_reuse'}
            assert all(r['task']==task and r['sample_id']==sample and r['seed']==0 for r in pair)
            assert all(pair[0][field]==pair[1][field] for field in generation_fields),(name,task)
            candidate=next(r for r in pair if r['arm']=='native_reuse')
            assert candidate['reuse_counts']['reused_layers']==candidate['reuse_counts']['reused_compactions']>0
            assert all(r['termination'] in ('eos','cap') and len(r['generated_token_ids'])<=r['max_gen'] for r in pair)
        assert inter['hot']==hot and inter['cold']==cold and len(inter['rows'])==9
        assert inter['native_reuse_launches']=={'calls':48,'layers':1536}
        for request,frames in [('A',13),('B',25),('C',13)]:
            rows=[r for r in inter['rows'] if r['request']==request]
            assert len(rows)==3 and {r['mode'] for r in rows}=={'gpu','cpu','interleaved_cpu'}
            for field in ('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters'):
                assert rows[0][field]==rows[1][field]==rows[2][field],(name,request,field)
            assert all(len(r['every_boundary_sha256'])==frames for r in rows)
        geometries[name]=dict(generation_pairs=12,tasks=6,recycled_requests=3,interleaved_frames_per_mode=51,exact_roots_released=15)
    suite=(B/'full_suite_v2.txt').read_text();assert '199 passed' in suite
    resource=json.loads((B/'verified_resources.json').read_text());assert resource['verified'] and not resource['candidate_ready_for_opt_in_cli']
    report=dict(verified=True,geometries=geometries,full_suite_tests=199,
        failed_counter_hook_run_preserved=True,candidate_ready_for_opt_in_cli=False,
        decision='Preservation passes; keep research-only because whole-resource gate remains failed/incomplete.',
        limitations='Reused development documents, seed zero, one model, serialized requests. No independent quality, threaded serving/throughput, GPU-peak improvement or physical STT claim.',
        evidence_sha256={p:sha(B/p) for p in evidence+bundles+['integrity_plan.json','integrity_retry_plan.json','interleaved_shared.json','interleaved_shared.log','full_suite_v2.txt','verified_resources.json']},verifier_sha256=sha(Path(__file__)))
    output=B/'verified_integrity.json';assert not output.exists();output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
