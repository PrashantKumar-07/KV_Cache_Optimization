"""Read-only archived free-generation preservation proof."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=BASE/'verified_generation.json';assert not out.exists()
p=read(BASE/'generation_plan.json');f=read(BASE/'generation_implementation.json');r=read(BASE/'generation.json')
assert p['cost_proof_sha256']==sha(BASE/'verified_cost.json') and read(BASE/'verified_cost.json')['cpu_phase_controller_wholeloop_primary_passed']
assert f['plan_sha256']==sha(BASE/'generation_plan.json') and f['archive_sha256']==sha(BASE/'generation_source.tar.gz')
with tarfile.open(BASE/'generation_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert sources==f['sources'] and all(sha(ROOT/n)==h for n,h in sources.items())
assert sha(ROOT/p['manifest'])==p['manifest_sha256']==r['manifest_sha256']
assert read(BASE/'generation_job.json')['exit_code']==0 and r['passed'] and r['owned_leases_closed']
assert r['actual_control']=='ordinary_bulk_promotion' and r['actual_candidate']=='cpu_phase_controller' and len(r['rows'])==24
fields=['prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes']
phases=evictions=reclaims=0
for first,second in zip(r['rows'][::2],r['rows'][1::2]):
 assert first['task']==second['task'] and first['sample_id']==second['sample_id'] and first['seed']==second['seed']==0
 pair={x['arm']:x for x in [first,second]};a=pair['fused_bounded'];b=pair['native_reuse']
 for field in fields:assert a[field]==b[field],(a['sample_id'],field)
 c=b['reuse_counts'];assert c['cpu_promote_calls']==c['cpu_append_calls']==c['cpu_demote_calls'] and c['cpu_fallback_steps']==0
 phases+=c['cpu_demote_calls'];evictions+=c['cpu_evictions'];reclaims+=c['cpu_reclaims']
 assert b['append_fallback_updates']==0
assert phases>0 and evictions==32*phases
result=dict(verified=True,document_pairs=12,tasks=sorted(set(x['task'] for x in r['rows'])),native_CPU_phase_steps=phases,native_CPU_layer_transitions=evictions,native_CPU_reclaims=reclaims,exact_prediction_score_state_trajectory_counters_native_KV=True,owned_leases_hooks_released=True,ordinary_adoption=False,full_goal_complete=False,limits=p['limits'],evidence_sha256={n:sha(BASE/n) for n in ['generation_plan.json','generation_implementation.json','generation_source.tar.gz','generation.json','generation_job.json','generation.log','check_generation.py','verify_generation.py','verified_cost.json']})
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
