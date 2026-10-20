"""Independently verify frozen reuse screens; does not launch a model or GPU job."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--v2',action='store_true')
    args=parser.parse_args();base=args.directory
    suffix='_v2' if args.v2 else ''
    raw=json.loads((base/f'result{suffix}.json').read_text())
    frozen=json.loads((base/f'source{suffix}_manifest.json').read_text())
    assert raw['passed'] and len(raw['checks'])==4 and len(raw['rows'])==24
    assert sha(base/f'source{suffix}.tar.gz')==frozen['bundle_sha256']
    with tarfile.open(base/f'source{suffix}.tar.gz') as archive:
        for name,digest in raw['source_sha256'].items():
            assert digest==frozen['source_sha256'][name]
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
        bench=ast.parse(archive.extractfile('experiments/longbench_eval.py').read())
    cls=next(n for n in bench.body if isinstance(n,ast.ClassDef) and n.name=='TieredKVPolicy')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_visible_tuple')
    original=ast.dump(ast.Module(body=[method],type_ignores=[]))
    assert hashlib.sha256(original.encode()).hexdigest()==raw['original_visible_ast_sha256']
    prior=json.loads((ROOT/'results/hardware/fused_cli_integration_20261002/source_manifest.json').read_text())['source_sha256']
    for name in ('experiments/fused_hot_compaction.py','experiments/fused_eviction_policy.py',
        'experiments/native_hot_compact.cpp','experiments/native_hot_compact.cu','experiments/bounded_pinned_pool.py',
        'experiments/longbench_eval.py','src/tiered_kv_cache.py'):
        assert frozen['source_sha256'][name]==prior[name]
    expected=json.loads((ROOT/'results/hardware/fused_eviction_model_20261002/verified_report.json').read_text())
    assert raw['binary_sha256']==expected['binary_sha256']
    assert [r['kept_rows'] for r in raw['checks']]==[64,32,64,4]
    assert all(r['layers']==4 and r['exact_bits'] and r['tail_untouched'] and r['visible_aliases_raw'] for r in raw['checks'])
    geometries={}
    for name,hot in (('shared',991),('default',1024)):
        rows=[r for r in raw['rows'] if r['geometry']==name]
        assert len(rows)==12
        assert [r['arm'] for r in rows]==['control','reuse','reuse','control']*3
        summary={};blocks={}
        for arm in ('control','reuse'):
            chosen=[r for r in rows if r['arm']==arm]
            assert all(r['hot']==hot and len(r['warmup_ms'])==4 and len(r['step_ms'])==16 and
                len(r['peak_incremental_bytes'])==16 for r in chosen)
            values=[v for r in chosen for v in r['step_ms']];assert len(values)==96
            assert all(v>0 for v in values)
            summary[arm]=dict(median_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),
                maximum_incremental_gpu_bytes=max(v for r in chosen for v in r['peak_incremental_bytes']))
            blocks[arm]=[statistics.median(v for r in chosen if r['block']==b for v in r['step_ms']) for b in range(3)]
        a,b=summary['control'],summary['reuse']
        saving=a['median_ms']-b['median_ms']
        guard=b['median_ms']<=1.02*a['median_ms'] and b['p95_ms']<=1.02*a['p95_ms']
        smaller=b['maximum_incremental_gpu_bytes']<a['maximum_incremental_gpu_bytes']
        calculated=dict(summary=summary,median_saving_ms=saving,charged_4ms_speed_screen_passed=saving>=4,
            storage_candidate_regression_guard=guard,incremental_storage_reduced=smaller)
        assert calculated==raw['geometries'][name]
        geometries[name]=dict(**calculated,block_medians=blocks,advance_storage_candidate=guard and smaller)
    if raw.get('native_backend',False):
        assert raw['reuse_binary_sha256']==sha(Path('/data/nishant/tieredkv_external/reuse_visible_build_20261002/tieredkv_reuse_hot_v1.so'))
        if args.v2:
            v1=json.loads((base/'result.json').read_text())
            old=json.loads((base/'source_manifest.json').read_text())['source_sha256']
            assert v1['reuse_binary_sha256']==raw['reuse_binary_sha256']
            for name in ('experiments/native_hot_compact.cpp','experiments/native_reuse_hot.cu'):
                assert old[name]==frozen['source_sha256'][name]
    result=dict(verified=True,geometries=geometries,native_backend=raw.get('native_backend',False),
        immutable_inputs_and_old_shader_verified=True,exact_gpu_pattern_and_alias_cases=4,
        advance_to_model_storage_gates=all(r['advance_storage_candidate'] for r in geometries.values()),
        any_new_4ms_speed_gate_passed=any(r['charged_4ms_speed_screen_passed'] for r in geometries.values()),
        reuse_binary_sha256=raw.get('reuse_binary_sha256'),
        result_sha256=sha(base/f'result{suffix}.json'),source_bundle_sha256=sha(base/f'source{suffix}.tar.gz'),
        verifier_sha256=sha(Path(__file__)),limitations=raw['limitations'])
    output=base/f'verified_report{suffix}.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
