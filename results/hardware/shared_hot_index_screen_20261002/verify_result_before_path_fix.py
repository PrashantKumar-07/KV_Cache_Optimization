"""Verify archived control extraction and recompute charged screen decision."""
import ast,hashlib,json,statistics,tarfile
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((P/'result.json').read_text());m=json.loads((P/'source_manifest.json').read_text())
assert r['passed'] and len(r['rows'])==12 and len(r['checks'])==12 and all(x['exact'] for x in r['checks'])
assert sha(P/'source.tar.gz')==m['bundle_sha256']
with tarfile.open(P/'source.tar.gz') as f:
 src={name:f.extractfile(name).read() for name in m['source_sha256']}
for name,value in src.items():assert hashlib.sha256(value).hexdigest()==m['source_sha256'][name]==r['source_sha256'][name]
methods={n.name:n for n in ast.walk(ast.parse(src['src/tiered_kv_cache.py'])) if isinstance(n,ast.FunctionDef)}
node=next(n for n in methods['evict_and_demote'].body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='victims')
function=ast.FunctionDef(name='compact',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='victims')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[ast.Assign(targets=[ast.Name(id='cfg',ctx=ast.Store())],value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='cfg',ctx=ast.Load())),node],decorator_list=[])
module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));assert hashlib.sha256(ast.dump(module).encode()).hexdigest()==r['reference_ast_sha256']
assert [x['arm'] for x in r['rows']]==['control','shared_index','shared_index','control']*3
summary={}
for arm in ('control','shared_index'):
 steps=[v for row in r['rows'] if row['arm']==arm for v in row['step_ms']];assert len(steps)==96
 summary[arm]={'median_ms':statistics.median(steps),'p95_ms':float(np.percentile(steps,95))}
saving=summary['control']['median_ms']-summary['shared_index']['median_ms'];assert abs(saving-r['median_saving_ms'])<1e-9
blocks=[{arm:statistics.median(v for row in r['rows'] if row['block']==block and row['arm']==arm for v in row['step_ms']) for arm in summary} for block in range(3)]
advance=saving>=4 and all(b['shared_index']<b['control'] for b in blocks) and summary['shared_index']['p95_ms']<=1.02*summary['control']['p95_ms'];assert advance==r['advance_to_model']
out=dict(source_and_exact_production_ast_and_gate_verified=True,summary=summary,blocks=blocks,median_saving_ms=saving,advance_to_model=advance,decision='reject: below4mschargedstageheadroom',raw_sha256=sha(P/'result.json'),source_bundle_sha256=m['bundle_sha256'],limitations=r['limitations'])
q=P/'verified_report.json';assert not q.exists();q.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
