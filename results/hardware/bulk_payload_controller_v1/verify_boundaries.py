import hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=read(B/'boundaries_plan.json');m=read(B/'boundaries_implementation.json');j=read(B/'boundaries_job.json')
assert j['status']=='finished' and j['exit_code']==0
assert sha(B/'verified_cost.json')==p['cost_proof_sha256'] and sha(B/'boundaries_plan.json')==m['plan_sha256'] and sha(B/'boundaries_source.tar.gz')==m['archive_sha256']
with tarfile.open(B/'boundaries_source.tar.gz') as t:
 for f,h in m['sources'].items():assert sha(R/f)==hashlib.sha256(t.extractfile(f).read()).hexdigest()==h
rows=[];evidence={}
for case in p['cases']:
 f=B/('boundary_'+case+'.json');d=read(f);assert d['passed'] and d['owned_leases_closed'];a,c=d['rows']
 assert a['trials'][0]['predicted_token_ids']==c['trials'][0]['predicted_token_ids'] and len(c['trials'][0]['predicted_token_ids'])==50
 for k in ['state_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','native_h2d_calls','native_d2h_calls']:assert a[k]==c[k],(case,k)
 n=c['phase_counts'];assert n['bulk_steps']+n['bulk_fallback_steps']==50
 if case in ['exclusive','write_aware_fallback']:assert n['bulk_steps']==0
 if case=='write_aware_fallback':assert n['cpu_append_calls']==0 and n['cpu_fallback_steps']==50
 else:assert n['cpu_append_calls']==50 and n['cpu_fallback_steps']==0
 rows.append(dict(case=case,**n));evidence[f.name]=sha(f)
out=B/'verified_boundaries.json';assert not out.exists();out.write_text(json.dumps(dict(verified=True,passed=True,cases=rows,evidence_sha256=evidence,exact_frozen_predictions_visible_state_counters_native_KV=True,limits=p['limits'],full_goal_complete=False),indent=2)+'\n');print(json.dumps(rows))
