"""Verify cached-loader mathematical preservation and ordinary CLI integration."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes=json.loads((BASE/'source_manifest.json').read_text())['source_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        sources={name:archive.extractfile(name).read() for name in hashes}
    for name,digest in hashes.items():assert hashlib.sha256(sources[name]).hexdigest()==digest
    previous=ROOT/'results/hardware/fused_eviction_stress_20261002'
    stress=json.loads((previous/'verified_report.json').read_text())
    primary=json.loads((ROOT/'results/hardware/fused_eviction_model_20261002/verified_report.json').read_text())
    assert stress['verified'] and primary['advance_to_mandatory_stress']
    old_hashes=json.loads((previous/'source_manifest.json').read_text())['source_sha256']
    for name in ('experiments/fused_eviction_policy.py','experiments/native_hot_compact.cpp',
        'experiments/native_hot_compact.cu','src/tiered_kv_cache.py','experiments/batched_eviction_policy.py'):
        assert hashes[name]==old_hashes[name]
    with tarfile.open(previous/'source.tar.gz') as archive:
        old=ast.parse(archive.extractfile('experiments/fused_hot_compaction.py').read())
    current=ast.parse(sources['experiments/fused_hot_compaction.py'])
    for name in ('gather','compact_group'):
        assert ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)) == ast.dump(
            next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name))
    control=json.loads((BASE/'cli_control.json').read_text())
    candidate=json.loads((BASE/'cli_candidate.json').read_text())
    for raw,fused in ((control,False),(candidate,True)):
        config=raw['config']
        assert config['cpu_fused_compaction']==fused and not config['cpu_bounded_pool']
        assert config['cpu_batched_scores'] and config['cpu_direct_shared'] and config['real_query_prefill']
        assert config['strategy']=='lru' and config['tiered_vram_budget']==1024 and config['tiered_stt_budget']==2048
        assert config['selector_sha256']==hashes['experiments/evaluate_query_selection.py']
    assert candidate['config']['fused_binary_sha256']==primary['binary_sha256']
    for name,digest in candidate['config']['fused_backend_sha256'].items():
        assert hashes[f'experiments/{name}']==digest
    assert candidate['config']['dataset_sha256']==control['config']['dataset_sha256']
    a=control['results']['qasper']['tieredkv'];b=candidate['results']['qasper']['tieredkv']
    assert a['n']==b['n']==1 and a['score']==b['score']
    x,y=a['samples'][0],b['samples'][0]
    for field in ('sample_id','prediction','score','prompt_tokens','generated_tokens','final_resident_kv_payload_bytes',
        'final_hot_kv_payload_bytes','final_cpu_kv_occupied_bytes','final_cpu_kv_reserved_bytes',
        'submitted_h2d_bytes','submitted_d2h_bytes'):
        assert x[field]==y[field],field
    assert y['fused_compactions']>0 and y['final_cpu_kv_reserved_bytes']==402784256
    for field in ('promoted','demoted','paid_writes','writes_saved','dropped','decode_steps'):
        assert a['cache_stats'][field]==b['cache_stats'][field],field
    invalid=json.loads((BASE/'invalid_flags.json').read_text())
    assert len(invalid)==5 and all(r['exit_code']==2 and r['preload_rejection_verified'] for r in invalid)
    commands=json.loads((BASE/'run_commands.json').read_text())
    assert len(commands)==2 and all(r['returncode']==0 and not r['ninja_on_path'] for r in commands)
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    report=dict(verified=True,cpu_gather_compaction_ast_unchanged=True,shader_binary_and_eviction_source_unchanged=True,
        cached_native_loader_without_ninja=True,cli_matching_sample=True,fused_compactions=y['fused_compactions'],
        invalid_preload_flag_cases=5,binary_sha256=primary['binary_sha256'],
        primary_report_sha256=sha(ROOT/'results/hardware/fused_eviction_model_20261002/verified_report.json'),
        stress_report_sha256=sha(previous/'verified_report.json'),
        source_archive_sha256=sha(BASE/'source.tar.gz'),verifier_sha256=sha(Path(__file__)),
        evidence_sha256={name:sha(BASE/name) for name in ('cli_control.json','cli_candidate.json','invalid_flags.json','run_commands.json')},
        limitations='CLI optin preserves one development sample; broad exactgeneration/stress in linked immutable evidence.11.96% speed refers to16K fixedteacherstream macrobenchmark with original history/synchronization. No new perCLI speed/quality/SnapKV/physicalSTT/resources/serving superiority. Combinedbounded/fused explicitly unverified and rejected.')
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
