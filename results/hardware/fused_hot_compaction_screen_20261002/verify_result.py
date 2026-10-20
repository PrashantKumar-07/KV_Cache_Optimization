"""Audit frozen compaction sources, binary, exact checks and charged timings."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

BASE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix',default='')
    args=parser.parse_args()
    suffix=args.suffix
    path=BASE/f'result{suffix}.json'
    raw=json.loads(path.read_text())
    frozen=json.loads((BASE/f'source{suffix}_manifest.json').read_text())['source_sha256']
    archive_path=BASE/f'source{suffix}.tar.gz'
    with tarfile.open(archive_path) as archive:
        for name,digest in raw['source_sha256'].items():
            assert frozen[name]==digest
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
        core=ast.parse(archive.extractfile('src/tiered_kv_cache.py').read())
    cls=next(n for n in core.body if isinstance(n,ast.ClassDef) and n.name=='TieredKVCache')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='evict_and_demote')
    node=next(n for n in method.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='victims')
    function=ast.FunctionDef(name='compact',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='victims')],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id='cfg',ctx=ast.Store())],value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='cfg',ctx=ast.Load())),copy.deepcopy(node)],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    assert hashlib.sha256(ast.dump(module).encode()).hexdigest()==raw['reference_ast_sha256']
    binary=Path('/data/nishant/tieredkv_external/fused_hot_compact_build_20261002/tieredkv_hot_compact_v1.so')
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==raw['binary_sha256']
    assert raw['passed'] and len(raw['checks'])==13 and all(r['exact'] for r in raw['checks'])
    assert len(raw['rows'])==12
    blocks=[]
    for block in range(3):
        rows=[r for r in raw['rows'] if r['block']==block]
        assert [r['arm'] for r in rows]==['control','fused','fused','control']
        for row in rows:
            assert len(row['warmup_ms'])==4 and len(row['step_ms'])==16
            assert all(v>0 for v in row['warmup_ms']+row['step_ms'])
        blocks.append({arm:statistics.median(v for r in rows if r['arm']==arm for v in r['step_ms']) for arm in ('control','fused')})
    summary={arm:dict(median_ms=statistics.median(v for r in raw['rows'] if r['arm']==arm for v in r['step_ms']),
        p95_ms=float(np.percentile([v for r in raw['rows'] if r['arm']==arm for v in r['step_ms']],95))) for arm in ('control','fused')}
    saving=summary['control']['median_ms']-summary['fused']['median_ms']
    advance=saving>=4 and all(b['fused']<b['control'] for b in blocks) and summary['fused']['p95_ms']<=1.02*summary['control']['p95_ms']
    assert summary==raw['summary'] and blocks==raw['blocks']
    assert saving==raw['median_saving_ms'] and advance==raw['advance_to_model']
    report=dict(verified=True,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        binary_sha256=raw['binary_sha256'],production_ast_verified=True,bitwise_cases=13,
        measured_steps_per_arm=96,summary=summary,blocks=blocks,median_saving_ms=saving,
        advance_to_model=advance,limitations=raw['limitations'])
    output=BASE/f'verified_report{suffix}.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
