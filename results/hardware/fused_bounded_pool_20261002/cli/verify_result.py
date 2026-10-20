"""Verify composed CLI source/data identity, reservation and output parity."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    frozen=json.loads((BASE/'source_manifest.json').read_text())
    hashes=frozen['source_sha256']
    assert sha(BASE/'source.tar.gz')==frozen['bundle_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for name,digest in hashes.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
    stress=json.loads((BASE.parent/'verified_stress.json').read_text())
    assert stress['verified'] and stress['advance_to_cli_combination']
    plan=json.loads((BASE/'plan.json').read_text())
    assert plan['stress_sha256']==sha(BASE.parent/'verified_stress.json')
    assert plan['systems_sha256']==sha(BASE.parent/'verified_systems.json')
    source=json.loads((BASE.parent/'source_stress_manifest.json').read_text())['source_sha256']
    for name in ('fused_bounded_pool.py','bounded_pinned_pool.py','fused_eviction_policy.py',
                 'fused_hot_compaction.py','native_hot_compact.cpp','native_hot_compact.cu'):
        assert hashes[f'experiments/{name}']==source[f'experiments/{name}']
    outcomes={}
    for name,(hot,cold) in plan['geometries'].items():
        rows={arm:json.loads((BASE/f'{name}_{arm}.json').read_text()) for arm in plan['arms']}
        for arm,raw in rows.items():
            cfg=raw['config'];combined=arm=='combined'
            assert cfg['cpu_fused_compaction'] and cfg['cpu_bounded_pool']==combined
            assert cfg['cpu_direct_shared'] and cfg['cpu_batched_scores'] and cfg['real_query_prefill']
            assert cfg['strategy']=='lru' and cfg['tiered_vram_budget']==hot and cfg['tiered_stt_budget']==cold
            assert cfg['selector_sha256']==hashes['experiments/evaluate_query_selection.py']
            assert cfg['fused_binary_sha256']==json.loads((BASE.parent/f'parity_{name}.json').read_text())['fused_binary_sha256']
            for path,digest in cfg['fused_backend_sha256'].items():assert hashes[f'experiments/{path}']==digest
            if combined:
                assert cfg['fused_bounded_backend_sha256']==hashes['experiments/fused_bounded_pool.py']
                assert cfg['bounded_backend_sha256']==hashes['experiments/bounded_pinned_pool.py']
                assert cfg['bounded_max_decode_append']==32
        assert rows['fused']['config']['dataset_sha256']==rows['combined']['config']['dataset_sha256']
        a,b=[rows[arm]['results']['qasper']['tieredkv'] for arm in plan['arms']]
        assert a['n']==b['n']==1 and a['score']==b['score']
        x,y=a['samples'][0],b['samples'][0]
        for field in ('sample_id','prediction','score','prompt_tokens','generated_tokens',
            'final_resident_kv_payload_bytes','final_hot_kv_payload_bytes','final_cpu_kv_occupied_bytes',
            'submitted_h2d_bytes','submitted_d2h_bytes','fused_compactions','eager_compactions'):
            assert x[field]==y[field],(name,field)
        assert y['fused_compactions']>0
        assert x['final_cpu_kv_reserved_bytes']==(cold+hot+1)*131072
        assert y['final_cpu_kv_reserved_bytes']==(cold+64)*131072
        for field in ('promoted','demoted','paid_writes','writes_saved','dropped','decode_steps'):
            assert a['cache_stats'][field]==b['cache_stats'][field],(name,field)
        outcomes[name]=dict(score=y['score'],fused_compactions=y['fused_compactions'],
            root_request_bytes=[x['final_cpu_kv_reserved_bytes'],y['final_cpu_kv_reserved_bytes']],exact=True)
    invalid=json.loads((BASE/'invalid_flags.json').read_text())
    commands=json.loads((BASE/'run_commands.json').read_text())
    assert len(invalid)==5 and all(r['exit_code']==2 and r['preload_rejection_verified'] for r in invalid)
    assert len(commands)==4 and all(r['returncode']==0 and not r['ninja_on_path'] for r in commands)
    assert '168 passed' in (BASE/'final_suite.txt').read_text()
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    result=dict(verified=True,geometries=outcomes,invalid_preload_cases=5,full_suite_tests=168,
        native_shader_and_composition_unchanged=True,cached_binary_without_ninja=True,
        stress_sha256=sha(BASE.parent/'verified_stress.json'),source_bundle_sha256=sha(BASE/'source.tar.gz'),
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p.name!='verified_report.json'},
        suite_sha256=sha(BASE/'final_suite.txt'),verifier_sha256=sha(Path(__file__)),
        limitations='One reused development sample per geometry; wider exact-generation/isolation proof linked. No new quality superiority, CLI timing, allocator/RSS or physical-STT claim. Default policies unchanged.')
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
