import hashlib
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'experiments'))
from provenance import runtime_identity
p=Path(__file__).resolve().parent
r=json.loads((p/'result.json').read_text());assert r['passed'] and len(r['rows'])==9
assert r['runtime']['source_sha256']==runtime_identity()['source_sha256']
for path,h in r['source_sha256'].items():assert hashlib.sha256((p/Path(path).name).read_bytes()).hexdigest()==h
assert len(set(r['input_sha256'].values()))==3
assert hashlib.sha256((root/'results/remediation_v3/latency_prompt.txt').read_bytes()).hexdigest()==r['prompt_sha256']
rows={(x['mode'],x['request']):x for x in r['rows']};assert len(rows)==9
fields=('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters')
for request,steps in (('A',12),('B',24),('C',12)):
 for mode in ('gpu','cpu','interleaved_cpu'):
  x=rows[(mode,request)]
  assert len(x['predicted_token_ids'])==steps+1
  assert len(x['every_boundary_sha256'])==steps+1
  assert x['counters']['decode_steps']==steps
  for field in fields:assert x[field]==rows[('gpu',request)][field],(request,mode,field)
assert [(x['mode'],x['request']) for x in r['rows'][-3:]]==[('interleaved_cpu','A'),('interleaved_cpu','C'),('interleaved_cpu','B')]
report=dict(verified=True,raw_sha256=hashlib.sha256((p/'result.json').read_bytes()).hexdigest(),source_sha256=r['source_sha256'],requests=3,arms=9,exact_boundary_frames_per_schedule=51,isolated_gpu_cpu_and_interleaved_cpu_ids_fullstate_counters_equal=True,limitations=r['limitations'])
(p/'verified_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
