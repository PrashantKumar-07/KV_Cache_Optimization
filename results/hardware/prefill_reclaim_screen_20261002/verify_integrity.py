"""Check cleanup-only source transforms, exact generation and recycled roots."""
import ast,hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cm=json.loads((B/'integrity_manifest.json').read_text())
    assert sha(B/'integrity_source.tar.gz')==cm['source_bundle_sha256']
    with tarfile.open(B/'integrity_source.tar.gz') as a:
        contents={m.name:a.extractfile(m).read() for m in a.getmembers() if m.isfile()}
    manifest={p:hashlib.sha256(data).hexdigest() for p,data in contents.items()}
    assert manifest['experiments/check_prefill_reclaim_integrity.py']==cm['driver_sha256']
    control_plan=json.loads((B/'plan.json').read_text())
    assert sha(B/'control_source.tar.gz')==control_plan['control_source_sha256']
    assert sha(Path('results/hardware/exact_pinned_pool_20261002/verified_resources.json'))==control_plan['control_proof_sha256']
    with tarfile.open(B/'control_source.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile():
                assert contents[member.name]==archive.extractfile(member).read(), member.name
    plan=json.loads((B/'integrity_plan.json').read_text())
    assert sha(B/'verified_abba.json')==plan['abba_proof_sha256']
    macro=json.loads((B/'verified_abba.json').read_text());assert macro['verified'] and macro['advance_to_mandatory_integrity']
    functions=ast.parse(contents['experiments/longbench_eval.py']).body
    function=next(n for n in functions if isinstance(n,ast.FunctionDef) and n.name=='generate_with_policy')
    tree=ast.Module(body=[function],type_ignores=[])
    original=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    loops=[i for i,n in enumerate(function.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step']
    assert len(loops)==1
    function.body[loops[0]:loops[0]]=ast.parse("del out\nif str(device).startswith('cuda'):\n    torch.cuda.empty_cache()").body
    candidate=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    fields=('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
        'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes')
    counts={};evidence=[]
    for name,hot,cold in [('shared',991,1090),('default',1024,2048)]:
        data=[]
        for kind,expected_roots in [('generation',24),('interleaved',3)]:
            filename=f'{kind}_{name}.json';evidence.append(filename)
            raw=json.loads((B/filename).read_text());data.append(raw)
            assert raw['passed'] and not raw.get('error') and raw['prefill_cleanup']
            assert all(manifest[p]==v for p,v in raw['cleanup_source_sha256'].items())
            stats=raw['exact_allocation_stats'];assert stats['live_allocations']==stats['live_requested_bytes']==0
            allocations={};released=set()
            for event in stats['events']:
                token=event['token'];assert event['bytes']==(cold+64)*131072
                if event['kind']=='allocate':
                    assert token not in allocations;allocations[token]=event['bytes']
                else:
                    assert event['kind']=='release' and token in allocations and token not in released
                    released.add(token)
            assert released==set(allocations) and len(allocations)==expected_roots
        gen,inter=data
        assert gen['arm_mapping']=={'fused_bounded':'exact_reuse_control','native_reuse':'exact_reuse_cleanup'}
        assert gen['generation_cleanup_ast']==dict(original_ast_sha256=original,candidate_ast_sha256=candidate,only_predecode_cleanup_changed=True)
        mp=Path(f'results/hardware/fused_bounded_pool_20261002/generation_{name}/generation_manifest.json')
        m=json.loads(mp.read_text());assert sha(mp)==gen['manifest_sha256'] and m['seeds']==[0]
        expected=[(task,sample) for task,entry in m['tasks'].items() for sample in entry['selected_sample_ids']]
        assert len(expected)==12 and len(gen['rows'])==24 and gen['hot']==hot and gen['cold']==cold
        for index,(task,sample) in enumerate(expected):
            pair=gen['rows'][index*2:index*2+2]
            assert {row['arm'] for row in pair}=={'fused_bounded','native_reuse'}
            assert all(row['task']==task and row['sample_id']==sample and row['seed']==0 for row in pair)
            assert all(pair[0][field]==pair[1][field] for field in fields),(name,task)
            for row in pair:
                assert row['reserved_host_tensor_bytes']==(cold+64)*131072
                assert row['termination'] in ('cap','eos') and len(row['generated_token_ids'])<=row['max_gen']
        assert inter['hot']==hot and inter['cold']==cold and len(inter['rows'])==9
        assert inter['native_reuse_launches']==dict(calls=48,layers=1536)
        for request,frames in [('A',13),('B',25),('C',13)]:
            rows=[row for row in inter['rows'] if row['request']==request]
            assert len(rows)==3 and {row['mode'] for row in rows}=={'gpu','cpu','interleaved_cpu'}
            for field in ('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters'):
                assert rows[0][field]==rows[1][field]==rows[2][field],(name,request,field)
            assert all(len(row['every_boundary_sha256'])==frames for row in rows)
        counts[name]=dict(generation_pairs=12,tasks=6,recycled_requests=3,boundary_frames_per_mode=51,
            exact_roots_allocated_and_released=27)
    assert '199 passed' in (B/'full_suite.txt').read_text()
    report=dict(verified=True,geometries=counts,full_suite_tests=199,
        preserved_policy_and_native_code=True,cleanup_only_generator_ast_verified=True,candidate_ready_for_cli=False,
        decision='Preservation and decode-reserved primary pass; keep caller cleanup and exact-root candidate research-only because complete actual-resource gate remains failed/incomplete.',
        limitations='One model/greedy seed/reused documents and serialized dispatch. Same policy/native code as the earlier tested lifetime/saturation sources. CUDA allocator cache is trimmed once per request, not every decode step. No threaded serving/throughput, independent quality, full model-wide active peak or physical-STT claim.',
        evidence_sha256={p:sha(B/p) for p in evidence+['control_source.tar.gz','plan.json','integrity_source.tar.gz','integrity_manifest.json','integrity_plan.json','verified_abba.json','full_suite.txt']},verifier_sha256=sha(Path(__file__)))
    output=B/'verified_integrity.json';assert not output.exists();output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
