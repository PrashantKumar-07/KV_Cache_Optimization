"""Read-only frozen current-capacity development generation preservation."""
import hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=read(B/'generation_plan.json');m=read(B/'generation_implementation.json');d=read(B/'generation.json')
assert sha(B/'verified_cost.json')==p['cost_proof_sha256'] and sha(B/'verified_stress.json')==p['stress_proof_sha256']
assert sha(R/'experiments/check_reuse_integrity.py')==p['helper_sha256']
assert sha(B/'generation_plan.json')==m['plan_sha256'] and sha(B/'generation_source.tar.gz')==m['archive_sha256']
with tarfile.open(B/'generation_source.tar.gz') as t:
 for f,h in m['sources'].items():assert sha(R/f)==hashlib.sha256(t.extractfile(f).read()).hexdigest()==h
assert sha(B/'generation_manifest.json')==p['manifest_sha256']==d['manifest_sha256']
j=read(B/'generation_job.json');assert j['status']=='finished' and j['exit_code']==0
assert d['passed'] and d['owned_leases_closed'] and len(d['rows'])==24
assert d['actual_control']=='cpu_phase' and d['actual_candidate']=='bulk_payload_controller'
fields=['prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes']
steps=bulk=fallback=reclaims=0
for first,second in zip(d['rows'][::2],d['rows'][1::2]):
 assert first['task']==second['task'] and first['sample_id']==second['sample_id'] and first['seed']==second['seed']==0
 pair={x['arm']:x for x in [first,second]};a=pair['fused_bounded'];c=pair['native_reuse']
 for k in fields:assert a[k]==c[k],(a['sample_id'],k)
 for row in [a,c]:assert row['append_fallback_updates']==0
 n=c['reuse_counts'];assert n['cpu_fallback_steps']==0 and n['cpu_append_calls']==n['bulk_steps']+n['bulk_fallback_steps']
 assert n['cpu_append_calls']>0
 steps+=n['cpu_append_calls'];bulk+=n['bulk_steps'];fallback+=n['bulk_fallback_steps'];reclaims+=n['cpu_reclaims']
out=B/'verified_generation.json';assert not out.exists();out.write_text(json.dumps(dict(verified=True,passed=True,document_pairs=12,tasks=sorted(set(x['task'] for x in d['rows'])),cpu_phase_steps=steps,bulk_steps=bulk,bulk_fallback_steps=fallback,cold_reclaims=reclaims,exact_prediction_score_state_trajectory_counters_native_KV=True,owned_leases_hooks_released=True,evidence_sha256={f:sha(B/f) for f in ['generation_plan.json','generation_implementation.json','generation_source.tar.gz','generation_manifest.json','generation.json','generation_job.json']},limits='Reused development documents, one model/seed; preservation only. No independent quality/speed/concurrency/physical STT claim.',ordinary_adoption=False,full_goal_complete=False),indent=2)+'\n');print(steps,bulk,fallback,reclaims)
