import hashlib,json,statistics
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
r=json.loads((p/'result_v3.json').read_text());assert r['passed']
assert hashlib.sha256((p/'probe_executed_v3.py').read_bytes()).hexdigest()==r['source_sha256']['experiments/probe_protected_mask_numpy.py']
assert hashlib.sha256(Path('src/tiered_kv_cache.py').read_bytes()).hexdigest()==r['source_sha256']['src/tiered_kv_cache.py']
assert len(r['checks'])==36 and sum(x.get('exact_rejection',False) for x in r['checks'])==3
assert len(r['rows'])==12
assert [(x['block'],x['arm']) for x in r['rows']]==[(b,a) for b in range(3) for a in ('baseline','numpy','numpy','baseline')]
assert all(len(x['step_ms'])==16 and len(x['warmup_ms'])==4 for x in r['rows'])
s={a:dict(median_ms=statistics.median(v for x in r['rows'] if x['arm']==a for v in x['step_ms']),p95_ms=float(np.percentile([v for x in r['rows'] if x['arm']==a for v in x['step_ms']],95))) for a in ('baseline','numpy')}
assert s==r['summary']
blocks=[{a:statistics.median(v for x in r['rows'] if x['block']==b and x['arm']==a for v in x['step_ms']) for a in s} for b in range(3)]
assert blocks==r['blocks']
saving=s['baseline']['median_ms']-s['numpy']['median_ms'];assert saving==r['median_saving_ms']
gate=saving>=4 and all(b['numpy']<b['baseline'] for b in blocks) and s['numpy']['p95_ms']<=1.02*s['baseline']['p95_ms']
assert r['advance_to_model']==gate==False
report=dict(verified=True,raw_sha256=hashlib.sha256((p/'result_v3.json').read_bytes()).hexdigest(),mask_or_input_rejection_parity_cases=36,summary=s,median_saving_ms=saving,advance_to_model=False,decision='Reject under preregistered4msheadroom; no cache change/model trial',limitations=r['limitations'])
(p/'verified_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
