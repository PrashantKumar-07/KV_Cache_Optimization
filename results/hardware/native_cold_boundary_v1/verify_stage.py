import hashlib,json,tarfile
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=read(B/'plan.json');m=read(B/'gpu_implementation.json');d=read(B/'gpu_stage.json');j=read(B/'gpu_job.json')
assert j['status']=='finished' and j['exit_code']==0 and read(B/'tests_job.json')['suite_passed']==516
assert sha(B/'plan.json')==m['plan_sha256']==d['plan_sha256'] and sha(B/'gpu_source.tar.gz')==m['archive_sha256']
with tarfile.open(B/'gpu_source.tar.gz') as t:
 for f,h in m['sources'].items():assert sha(R/f)==hashlib.sha256(t.extractfile(f).read()).hexdigest()==h
assert sha(Path(d['binary_path']))==d['binary_sha256']
assert d['passed'] and d['nondefault_stream_handoff_exact'] and d['all_owned_hooks_leases_pool_released']
assert len(d['checks'])==4 and all(x['exact'] for x in d['checks']) and len(d['rejections'])==3 and all(x['atomic'] for x in d['rejections'])
assert len(d['rows'])==36 and len(d['summary'])==3
for summary in d['summary']:
 rows=[r for r in d['rows'] if (r['hot'],r['cold'],r['case'])==(summary['hot'],summary['cold'],summary['case'])];assert len(rows)==12
 for block in range(3):
  four=rows[block*4:block*4+4];assert [r['arm'] for r in four]==['current','native','native','current'] and all(r['block']==block and len(r['step_ms'])==20 and len(r['warmup_ms'])==4 for r in four)
 data={a:[v for r in rows if r['arm']==a for v in r['step_ms']] for a in ['current','native']};med={a:float(np.median(v)) for a,v in data.items()};p95={a:float(np.quantile(v,.95)) for a,v in data.items()}
 assert med==summary['median_ms'] and p95==summary['p95_ms'] and med['current']-med['native']==summary['saved_ms']
 assert summary['guards']==dict(saved_ms=summary['saved_ms']>=4,median=med['native']/med['current']<=.95,p95=p95['native']/p95['current']<=1.02)
 assert summary['passed']==all(summary['guards'].values())
passed=all(s['passed'] for s in d['summary']);assert passed==d['actual_stage_gate_passed']
out=B/'verified_stage.json';assert not out.exists();out.write_text(json.dumps(dict(verified=True,actual_stage_gate_passed=passed,summary=d['summary'],checks=d['checks'],atomic_rejections=d['rejections'],nondefault_stream_handoff_exact=True,measured_calls_per_arm_per_case=120,tests_passed=516,binary_path=d['binary_path'],binary_sha256=d['binary_sha256'],evidence_sha256={f:sha(B/f) for f in ['plan.json','gpu_implementation.json','gpu_source.tar.gz','gpu_stage.json','gpu_job.json','tests_job.json']},disposition='Advance to separately preregistered model integration, exact GPU replay/fallback/cleanup, fresh resources and macro gate.' if passed else 'Stop mechanism; no model expansion/adoption or favorable timing retry.',limits=p['limits'],full_goal_complete=False),indent=2)+'\n');print('Stage gate:',passed)
