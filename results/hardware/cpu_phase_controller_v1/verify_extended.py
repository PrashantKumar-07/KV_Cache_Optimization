"""Independent archived extended parity and actual CPU phase dispatch checks."""
import hashlib,json,tarfile,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
kind=sys.argv[1];out=BASE/('verified_'+kind+'.json');assert not out.exists()
f=read(BASE/(kind+'_implementation.json'));p=read(BASE/(kind+'_plan.json'));r=read(BASE/(kind+'.json'))
assert f['plan_sha256']==sha(BASE/(kind+'_plan.json')) and p['cost_proof_sha256']==sha(BASE/'verified_cost.json')
assert read(BASE/'verified_cost.json')['cpu_phase_controller_wholeloop_primary_passed']
assert f['archive_sha256']==sha(BASE/(kind+'_source.tar.gz'))
with tarfile.open(BASE/(kind+'_source.tar.gz')) as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert sources==f['sources'] and all(sha(ROOT/n)==h for n,h in sources.items())
assert read(BASE/(kind+'_job.json'))['exit_code']==0 and r['passed'] and r['owned_leases_closed']
assert len(r['rows'])==(4 if kind=='wider' else 2)
for a,b in zip(r['rows'][::2],r['rows'][1::2]):
 assert a['context']==b['context']
 for field in ['state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_h2d_calls','native_d2h_calls','per_layer_initial_d2h_calls']:assert a[field]==b[field],field
 assert len(a['every_visible_state_sha256'])==51
 for x,y in zip(a['trials'],b['trials']):
  for field in ['predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps']:assert x[field]==y[field],field
  assert x['decode_steps']==50 and len(x['predicted_token_ids'])==50
assert len(r['append_summaries'])==(2 if kind=='wider' else 1)
for c in r['append_summaries']:
 for n,v in dict(fast_updates=1600,fallback_updates=0,pending_released=True,cpu_promote_calls=50,cpu_append_calls=50,cpu_demote_calls=50,cpu_fallback_steps=0,cpu_evictions=1600,controller_plan_calls=50,controller_fallback_plans=0,controller_backup_calls=0).items():assert c[n]==v,n
 if kind=='recycling':assert c['cpu_reclaims']>0
result=dict(verified=True,kind=kind,exact_model_and_native_KV_parity=True,contexts=[x['context'] for x in r['rows'][::2]],steps_per_context=50,cpu_counts=r['append_summaries'],full_goal_complete=False,ordinary_adoption=False,limits=p['limits'],evidence_sha256={n:sha(BASE/n) for n in [kind+'_plan.json',kind+'_implementation.json',kind+'_source.tar.gz',kind+'.json',kind+'_job.json',kind+'.log','verify_extended.py','verified_cost.json']})
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','cpu_counts']},indent=2))
