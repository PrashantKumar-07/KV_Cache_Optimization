import hashlib,json,statistics
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
output=p/'verified_report.json';assert not output.exists()
rows=[]
for raw,executed in (('result.json','probe_executed.py'),('combined.json','probe_combined_executed.py')):
 r=json.loads((p/raw).read_text());assert r['passed']
 assert hashlib.sha256((p/executed).read_bytes()).hexdigest()==r['source_sha256']['experiments/probe_batched_eviction_selection.py']
 assert hashlib.sha256(Path('src/tiered_kv_cache.py').read_bytes()).hexdigest()==r['source_sha256']['src/tiered_kv_cache.py']
 if 'experiments/probe_protected_mask_numpy.py' in r['source_sha256']:
  assert hashlib.sha256((p/'numpy_mask_executed.py').read_bytes()).hexdigest()==r['source_sha256']['experiments/probe_protected_mask_numpy.py']
 assert len(r['checks'])==7 and all(x['exact'] for x in r['checks'])
 assert len(r['rows'])==12 and [(x['block'],x['arm']) for x in r['rows']]==[(b,a) for b in range(3) for a in ('baseline','batched','batched','baseline')]
 assert all(len(x['warmup_ms'])==4 and len(x['step_ms'])==16 for x in r['rows'])
 s={a:dict(median_ms=statistics.median(v for x in r['rows'] if x['arm']==a for v in x['step_ms']),p95_ms=float(np.percentile([v for x in r['rows'] if x['arm']==a for v in x['step_ms']],95))) for a in ('baseline','batched')}
 blocks=[{a:statistics.median(v for x in r['rows'] if x['block']==b and x['arm']==a for v in x['step_ms']) for a in s} for b in range(3)]
 saving=s['baseline']['median_ms']-s['batched']['median_ms']
 gate=saving>=4 and all(b['batched']<b['baseline'] for b in blocks) and s['batched']['p95_ms']<=1.02*s['baseline']['p95_ms']
 assert s==r['summary'] and blocks==r['blocks'] and saving==r['median_saving_ms'] and gate==r['advance_to_model']
 rows.append(dict(raw=raw,raw_sha256=hashlib.sha256((p/raw).read_bytes()).hexdigest(),reference_ast_sha256=r['reference_ast_sha256'],summary=s,blocks=blocks,median_saving_ms=saving,advance_to_model=gate))
assert rows[0]['reference_ast_sha256']==rows[1]['reference_ast_sha256']
assert not rows[0]['advance_to_model'] and rows[1]['advance_to_model']
report=dict(verified=True,variants=rows,cache_source_unchanged=True,limitations='Synthetic fullselection stage afterpromotion/append, not preparingthose states or wholemodel execution. Initialnonfinitefixtureprotectedvaluesnotstrongcoverage; combinedfixtureexposesnonfiniteoutside sinks. Production cache/policy unchanged; headroomonly, notquality/speed/matchedresource claim.')
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
