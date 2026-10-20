import hashlib,json,tarfile
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=read(B/'screen_plan.json');m=read(B/'screen_implementation.json');d=read(B/'screen.json');j=read(B/'screen_job.json')
assert j['status']=='finished' and j['exit_code']==0 and d['passed'] and d['owned_hooks_leases_pool_released']
assert d['model_frozen_replay_exact'] and d['cache_visible_and_cpuowners_unchanged']
assert sha(R/p['reference'])==p['reference_sha256'];assert sha(B/'screen_plan.json')==m['plan_sha256']==d['plan_sha256'];assert sha(B/'screen_source.tar.gz')==m['archive_sha256']
with tarfile.open(B/'screen_source.tar.gz') as t:
 for f,h in m['sources'].items():assert sha(R/f)==hashlib.sha256(t.extractfile(f).read()).hexdigest()==h
assert len(d['rows'])==12
for block in range(3):
 rows=d['rows'][4*block:4*block+4];assert [r['arm'] for r in rows]==['current','zero_work','zero_work','current']
 assert all(r['block']==block and len(r['step_ms'])==20 and len(r['warmup_ms'])==4 for r in rows)
med={a:float(np.median([v for r in d['rows'] if r['arm']==a for v in r['step_ms']])) for a in ['current','zero_work']}
assert med==d['median_ms'] and med['current']-med['zero_work']==d['optimistic_saving_ms']
assert d['charged_h2d_calls']==144 and d['charged_h2d_bytes']==144*4*2**20 and d['measured_calls_per_arm']==120
out=B/'verified_screen.json';assert not out.exists();out.write_text(json.dumps(dict(verified=True,median_ms=med,optimistic_saving_ms=d['optimistic_saving_ms'],can_meet_4ms_screen=d['can_meet_4ms_screen'],native_transfer_calls_charged=144,native_transfer_bytes_charged=d['charged_h2d_bytes'],actual_frozen_model_initialization=True,state_and_visible_bits_and_cpuowners_preserved=True,limits=p['limits'],evidence_sha256={f:sha(B/f) for f in ['screen_plan.json','screen_implementation.json','screen_source.tar.gz','screen.json','screen_job.json']},full_goal_complete=False),indent=2)+'\n');print('Screen verified:',med)
