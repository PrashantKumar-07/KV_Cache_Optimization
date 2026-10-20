"""Independent archived model boundary, native/fallback dispatch verification."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
out=BASE/'verified_boundaries.json';assert not out.exists()
f=read(BASE/'gpu_implementation.json');p=read(BASE/'plan.json')
assert sha(BASE/'plan.json')==f['plan_sha256'] and sha(BASE/'gpu_source.tar.gz')==f['archive_sha256']
with tarfile.open(BASE/'gpu_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert sources==f['sources'] and all(sha(ROOT/n)==h for n,h in sources.items())
assert read(BASE/'gpu_job.json')['exit_code']==0
results={}
for name,options in p['cases'].items():
 r=read(BASE/(name+'.json'));assert r['passed'] and r['owned_leases_closed'] and r['boundary_case']==name and r['boundary_options']==options
 assert len(r['rows'])==2 and len(r['append_summaries'])==1
 a,b=r['rows'];assert a['context']==b['context']==4096
 for n in ['state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_h2d_calls','native_d2h_calls','per_layer_initial_d2h_calls']:assert a[n]==b[n],(name,n)
 assert len(a['every_visible_state_sha256'])==51
 for x,y in zip(a['trials'],b['trials']):
  for n in ['predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps']:assert x[n]==y[n],(name,n)
  assert x['decode_steps']==len(x['predicted_token_ids'])==50
 c=r['append_summaries'][0]
 for n,v in dict(fast_updates=1600,fallback_updates=0,pending_released=True,controller_plan_calls=50,controller_fallback_plans=0,cpu_owned_bytes=439648).items():assert c[n]==v,(name,n)
 if name!='write_aware_fallback':
  assert c['cpu_promote_calls']==c['cpu_append_calls']==c['cpu_demote_calls']==50
  assert c['cpu_fallback_steps']==0 and c['cpu_evictions']==1600 and c['cpu_reclaims']==576 and c['controller_backup_calls']==0
 else:
  assert c['cpu_promote_calls']==c['cpu_append_calls']==c['cpu_demote_calls']==c['cpu_evictions']==c['cpu_reclaims']==0
  assert c['cpu_fallback_steps']==50 and c['controller_backup_calls']==1600
 t=b['trials'][0]
 if name=='exclusive':assert t['promoted']==24832 and t['paid_writes']==t['demoted']==26432 and t['writes_saved']==0
 if name=='floor_over_capacity':assert t['promoted']==0 and t['paid_writes']==t['demoted']==1600
 results[name]=dict(exact=True,counts=c,policy_outcomes={n:t[n] for n in ['promoted','demoted','paid_writes','writes_saved']})
r=dict(verified=True,cases=results,exact_native_CPU_model_and_fallback_parity=True,model_decode_forwards=300,visibility_frames_per_arm=51,total_native_CPU_reclaims=1152,all_owned_jobs_terminal=True,owned_leases_hooks_released=True,full_goal_complete=False,ordinary_adoption=False,limits=p['limits'],evidence_sha256={n:sha(BASE/n) for n in ['plan.json','gpu_implementation.json','gpu_source.tar.gz','gpu_job.json','gpu.log','check_gpu.py','verify_boundaries.py',*[(name+'.json') for name in p['cases']]]})
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['cases','evidence_sha256']},indent=2))
