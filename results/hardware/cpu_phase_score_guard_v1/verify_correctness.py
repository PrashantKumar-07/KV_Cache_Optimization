"""Read-only archive, recorded replay and actual pooled dispatch verification."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
output=BASE/'verified_correctness.json';assert not output.exists()
frozen=read(BASE/'gpu_implementation.json');raw=read(BASE/'gpu.json')
assert frozen['plan_sha256']==sha(BASE/'plan.json') and frozen['archive_sha256']==sha(BASE/'gpu_source.tar.gz')
with tarfile.open(BASE/'gpu_source.tar.gz') as t:
 sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
assert read(BASE/'gpu_job.json')['exit_code']==0 and read(BASE/'tests_job.json')['exit_code']==0
assert '498 passed' in (BASE/'suite.log').read_text()
assert raw['passed'] and raw['owned_leases_closed'] and len(raw['rows'])==2
a,b=raw['rows'];assert a['context']==b['context']==4096
for f in ['state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_h2d_calls','native_d2h_calls','per_layer_initial_d2h_calls']:assert a[f]==b[f],f
assert len(a['every_visible_state_sha256'])==13
for x,y in zip(a['trials'],b['trials']):
 for f in ['predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps']:assert x[f]==y[f],f
 assert x['decode_steps']==12 and len(x['predicted_token_ids'])==12
assert len(raw['append_summaries'])==1
counts=raw['append_summaries'][0]
for f,v in dict(fast_updates=384,fallback_updates=0,pending_released=True,shared_transition_bindings=384,shared_immediate_compactions=0,batched_deferred_compactions=384,bounded_native_groups=0,controller_plan_calls=12,controller_fallback_plans=0,pooled_prepare_calls=12,pooled_compact_groups=48,pooled_owned_gpu_bytes=417792,pooled_descriptor_h2d_bytes=1549824,pooled_scratch_peak_bytes=32536512).items():assert counts[f]==v,f
for f,v in dict(cpu_promote_calls=12,cpu_append_calls=12,cpu_demote_calls=12,cpu_fallback_steps=0,cpu_evictions=384,cpu_reclaims=0,cpu_owned_bytes=1116896,controller_backup_calls=0).items():assert counts[f]==v,f
result=dict(verified=True,tests_passed=498,focused_tests=15,prediction_steps=12,visibility_frames=13,exact_native_kv=True,pooled_counts=counts,ready_for_resource=True,full_goal_complete=False,ordinary_adoption=False,limits='Single recorded 4K12 pair, no separately dumped logits. GPU arena bytes and scratch are components, not full-model peak. No speed, quality, faithful baseline or physical STT claim.',evidence_sha256={f:sha(BASE/f) for f in ['plan.json','gpu_implementation.json','gpu_source.tar.gz','gpu.json','gpu.log','gpu_job.json','suite.log','tests_job.json','verify_correctness.py']})
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
