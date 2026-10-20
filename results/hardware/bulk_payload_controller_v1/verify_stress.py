import hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
j=read(B/'stress_job.json');assert j['status']=='finished' and j['exit_code']==0
m=read(B/'stress_implementation.json');assert sha(B/'stress_plan.json')==m['plan_sha256'];assert sha(B/'stress_source.tar.gz')==m['archive_sha256']
with tarfile.open(B/'stress_source.tar.gz') as t:
 for f,h in m['sources'].items():assert sha(R/f)==hashlib.sha256(t.extractfile(f).read()).hexdigest()==h
p=read(B/'stress_plan.json');assert sha(B/'verified_cost.json')==p['cost_proof_sha256']
rows=[];evidence={}
for case in p['cases']:
 f=B/f"stress_{case['context']}_{case['decode']}_{case['cold']}.json";d=read(f);assert d['passed'] and d['owned_leases_closed']
 a,c=d['rows'];assert a['trials'][0]['predicted_token_ids']==c['trials'][0]['predicted_token_ids'];assert a['state_sha256']==c['state_sha256'] and a['every_visible_state_sha256']==c['every_visible_state_sha256']
 for k in ['explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','native_h2d_calls','native_d2h_calls']:assert a[k]==c[k],k
 n=c['phase_counts'];assert n['bulk_steps']+n['bulk_fallback_steps']==50 and n['cpu_append_calls']==50 and n['cpu_fallback_steps']==0
 assert n['cpu_reclaims']==a['phase_counts']['cpu_reclaims']
 if case['cold']==256:assert n['cpu_reclaims']>0 and n['bulk_fallback_steps']>0
 rows.append(dict(**case,**n));evidence[f.name]=sha(f)
out=B/'verified_stress.json';assert not out.exists();out.write_text(json.dumps(dict(verified=True,passed=True,cases=rows,evidence_sha256=evidence,limits='Exact teacher replay and physical-slot recycling preservation; no separately captured full-model logits, quality or stress latency primary.',full_goal_complete=False),indent=2)+'\n');print(json.dumps(rows))
