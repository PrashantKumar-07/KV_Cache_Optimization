"""Verify CLI native reuse adoption against frozen integrity and outputs."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def definition(source,name):
    tree=ast.parse(source)
    return ast.dump(next(node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name))

def main():
    plan=json.loads((BASE/'plan.json').read_text())
    proof=json.loads((BASE.parent/'verified_report.json').read_text())
    assert proof['verified'] and proof['candidate_ready_for_opt_in_cli']
    assert sha(BASE.parent/'verified_report.json')==plan['integrity_sha256']
    bundle=json.loads((BASE/'source_manifest.json').read_text())
    assert sha(BASE/'source.tar.gz')==bundle['bundle_sha256']
    hashes=bundle['source_sha256']
    with tarfile.open(BASE/'source.tar.gz') as current,tarfile.open(BASE.parent/'source.tar.gz') as previous:
        for name,digest in hashes.items():assert hashlib.sha256(current.extractfile(name).read()).hexdigest()==digest
        for name in ('src/tiered_kv_cache.py','experiments/reuse_visible_policy.py','experiments/fused_bounded_pool.py',
            'experiments/fused_eviction_policy.py','experiments/fused_hot_compaction.py','experiments/native_hot_compact.cpp',
            'experiments/native_reuse_hot.cu'):
            assert hashlib.sha256(previous.extractfile(name).read()).hexdigest()==hashes[name]
        name='experiments/native_reuse_visible.py'
        for symbol in ('gather_into','NativeReusePolicy'):
            assert definition(previous.extractfile(name).read().decode(),symbol)==definition(current.extractfile(name).read().decode(),symbol)
    outcomes={}
    for geometry,(hot,cold) in plan['geometries'].items():
        rows={arm:json.loads((BASE/f'{geometry}_{arm}.json').read_text()) for arm in plan['arms']}
        for arm,raw in rows.items():
            cfg=raw['config']
            assert cfg['cpu_reuse_visible']==(arm=='reuse')
            assert all(cfg[key] for key in ('cpu_fused_compaction','cpu_bounded_pool','cpu_direct_shared','cpu_batched_scores','real_query_prefill'))
            assert cfg['tiered_vram_budget']==hot and cfg['tiered_stt_budget']==cold and cfg['strategy']=='lru'
            assert cfg['selector_sha256']==hashes['experiments/evaluate_query_selection.py']
            if arm=='reuse':
                assert cfg['reuse_binary_sha256']==proof['reuse_binary_sha256']
                assert all(hashes[f'experiments/{name}']==value for name,value in cfg['reuse_backend_sha256'].items())
            else:assert cfg['reuse_backend_sha256'] is None
        a,b=[rows[arm]['results']['qasper']['tieredkv'] for arm in plan['arms']]
        assert a['n']==b['n']==1 and a['score']==b['score']
        assert rows['control']['config']['dataset_sha256']==rows['reuse']['config']['dataset_sha256']
        x,y=a['samples'][0],b['samples'][0]
        for field in ('sample_id','prediction','score','prompt_tokens','generated_tokens','final_resident_kv_payload_bytes',
            'final_hot_kv_payload_bytes','final_cpu_kv_occupied_bytes','final_cpu_kv_reserved_bytes','submitted_h2d_bytes','submitted_d2h_bytes','eager_compactions'):
            assert x[field]==y[field],(geometry,field)
        assert y['reused_compactions']>0 and y['reuse_fallback_layers']==0
        assert x['fused_compactions']==y['reused_compactions']+y['fused_compactions']
        assert y['final_cpu_kv_reserved_bytes']==(cold+64)*131072
        for field in ('promoted','demoted','paid_writes','writes_saved','dropped','decode_steps'):
            assert a['cache_stats'][field]==b['cache_stats'][field],(geometry,field)
        historical=Path(f'results/hardware/fused_bounded_pool_20261002/cli/{geometry}_combined.json')
        old=json.loads(historical.read_text())['results']['qasper']['tieredkv']
        assert a['score']==old['score'] and x['prediction']==old['samples'][0]['prediction']
        outcomes[geometry]=dict(exact=True,reused_compactions=y['reused_compactions'],score=y['score'],root_bytes=y['final_cpu_kv_reserved_bytes'])
    invalid=json.loads((BASE/'invalid_flags.json').read_text());assert len(invalid)==7 and all(r['preload_rejection_verified'] and r['exit_code']==2 for r in invalid)
    commands=json.loads((BASE/'run_commands.json').read_text());assert len(commands)==4 and all(r['returncode']==0 and not r['ninja_on_path'] for r in commands)
    suite=BASE.parent/'integration_suite.txt';assert '187 passed' in suite.read_text()
    result=dict(verified=True,geometries=outcomes,invalid_preload_cases=7,full_suite_tests=187,cached_binary_without_ninja=True,
        native_shader_and_gather_ast_unchanged=True,ordinary_opt_in_flag='--cpu-reuse-visible',default_change=False,
        integrity_sha256=sha(BASE.parent/'verified_report.json'),source_bundle_sha256=sha(BASE/'source.tar.gz'),suite_sha256=sha(suite),
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p.name!='verified_report.json'},verifier_sha256=sha(Path(__file__)),
        limitations='Storage-only opt-in tradeoff; no new speed, model-wide memory, independent quality, physical STT or SnapKV superiority claim. One reused CLI sample per geometry with wider archived generation/native/recycling proofs. No threaded serving/throughput proof.')
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
