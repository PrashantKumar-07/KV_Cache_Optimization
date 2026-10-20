"""Read-only recheck that all cleanup generation rows reproduce frozen outputs."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
OLD=Path('results/hardware/exact_pinned_pool_20261002')
FIELDS=('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
 'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
 'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes')
evidence={}
for name in ('shared','default'):
 prior=json.loads((OLD/f'generation_{name}.json').read_text())
 current=json.loads((B/f'generation_{name}.json').read_text())
 assert prior['passed'] and current['passed']
 baseline={(r['task'],r['sample_id']):r for r in prior['rows'] if r['arm']=='native_reuse'}
 assert len(baseline)==12 and len(current['rows'])==24
 for row in current['rows']:
  assert all(row[f]==baseline[(row['task'],row['sample_id'])][f] for f in FIELDS)
 for root in (OLD,B):
  path=root/f'generation_{name}.json'
  evidence[str(path.relative_to(Path.cwd())) if path.is_absolute() else str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
report=json.loads((B/'verified_frozen_replay.json').read_text())
assert report['verified'] and report['generation_rows']==48 and report['frozen_exact_control_reproduced']
assert report['evidence_sha256']==evidence
print('48 generation rows reproduce frozen exact-root control across all 16 preservation fields')
