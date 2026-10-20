"""Audit ordinary opt-in flags, source identities, outputs and cached loading."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda name:json.loads((BASE/name).read_text())


def main():
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    plan=read('plan.json')
    assert sha(BASE/'source.tar.gz')==plan['source_bundle_sha256']
    for filename,key in (('verified_integrity.json','integrity_proof_sha256'),('verified_shared_timing.json','shared_timing_proof_sha256')):
        assert sha(BASE.parent/filename)==plan[key]
    with tarfile.open(BASE/'source.tar.gz') as archive:
        frozen={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in frozen.items()}
    assert sha(Path('experiments/evaluate_query_selection.py'))==hashes['experiments/evaluate_query_selection.py']
    # Descriptions changed after integrity; measured algorithm/class ASTs did not.
    with tarfile.open(BASE.parent/'integrity_source.tar.gz') as archive:
        for name in ('experiments/bulk_promotion_stage.py','experiments/bulk_promotion_policy.py'):
            measured=ast.parse(archive.extractfile(name).read());active=ast.parse(frozen[name])
            assert ast.dump(ast.Module(body=measured.body[1:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=active.body[1:],type_ignores=[]),include_attributes=False)
    native=json.loads((BASE.parent/'model_parity.json').read_text())['reuse_binary_sha256']
    fields=('sample_id','prediction','score','prompt_tokens','generated_tokens','final_resident_kv_payload_bytes',
        'final_hot_kv_payload_bytes','final_cpu_kv_occupied_bytes','final_cpu_kv_reserved_bytes','submitted_h2d_bytes',
        'submitted_d2h_bytes','eager_compactions','fused_compactions','reused_compactions','reuse_fallback_layers')
    outcomes={}
    for geometry,(hot,cold) in plan['geometries'].items():
        reports={arm:read(f'{geometry}_{arm}.json') for arm in plan['arms']}
        for arm,report in reports.items():
            cfg=report['config']
            assert cfg['cpu_bulk_promotion']==(arm=='bulk')
            assert all(cfg[flag] for flag in ('cpu_reuse_visible','cpu_fused_compaction','cpu_bounded_pool','cpu_direct_shared','cpu_batched_scores','real_query_prefill'))
            assert cfg['tiered_vram_budget']==hot and cfg['tiered_stt_budget']==cold and cfg['strategy']=='lru'
            assert cfg['selector_sha256']==hashes['experiments/evaluate_query_selection.py'] and cfg['reuse_binary_sha256']==native
            if arm=='bulk':
                for name,digest in cfg['bulk_promotion_backend_sha256'].items():
                    normalized=str(Path('experiments',name).resolve().relative_to(Path.cwd()))
                    assert digest==hashes[normalized]
            else:assert cfg['bulk_promotion_backend_sha256'] is None
        a,b=[reports[arm]['results']['qasper']['tieredkv'] for arm in plan['arms']]
        assert a['n']==b['n']==1 and a['score']==b['score']
        assert reports['control']['config']['dataset_sha256']==reports['bulk']['config']['dataset_sha256']
        x,y=a['samples'][0],b['samples'][0]
        for field in fields:assert x[field]==y[field],(geometry,field)
        assert y['reused_compactions']>0 and y['reuse_fallback_layers']==0
        assert y['final_cpu_kv_reserved_bytes']==(cold+64)*131072
        for field in ('promoted','demoted','paid_writes','writes_saved','dropped','decode_steps'):
            assert a['cache_stats'][field]==b['cache_stats'][field],(geometry,field)
        previous=json.loads(Path(f'results/hardware/reuse_visible_integrity_20261002/cli/{geometry}_reuse.json').read_text())['results']['qasper']['tieredkv']
        for field in fields:assert x[field]==previous['samples'][0][field],('prior ordinary replay',geometry,field)
        outcomes[geometry]=dict(exact=True,score=b['score'],reused_compactions=y['reused_compactions'],root_bytes=y['final_cpu_kv_reserved_bytes'])
    invalid=read('invalid_flags.json')
    assert len(invalid)==8 and all(r['preload_rejection_verified'] and r['exit_code']==2 for r in invalid)
    commands=read('run_commands.json')
    assert len(commands)==4 and all(r['returncode']==0 and not r['ninja_on_path'] for r in commands)
    suite=BASE.parent/'cli_suite.txt';assert '206 passed' in suite.read_text()
    proof=dict(verified=True,candidate_ready_for_opt_in_cli=True,ordinary_opt_in_flag='--cpu-bulk-promotion',
        default_change=False,geometries=outcomes,invalid_preload_cases=8,full_suite_tests=206,
        cached_binary_without_ninja=True,measured_algorithm_ast_unchanged=True,
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p!=output}|{
            'source.tar.gz':sha(BASE/'source.tar.gz'),'cli_suite.txt':sha(suite)},
        verifier_sha256=sha(Path(__file__)),limitations='One reused ordinary sample per geometry plus archived wider integrity/timing gates. CPU-control speed gains only; no quality/physicalSTT/actual-resource/SnapKV/serving-superiority claim.')
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(verified=True,geometries=outcomes,ordinary_opt_in_flag=proof['ordinary_opt_in_flag']),indent=2))


if __name__=='__main__':main()
