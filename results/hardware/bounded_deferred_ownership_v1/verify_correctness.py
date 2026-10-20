"""Read-only recordedpair/source proof; reserved append actually dispatched."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_correctness.json';assert not output.exists()
    plan=read(BASE/'plan.json');frozen=read(BASE/'gpu_implementation.json');raw=read(BASE/'gpu.json')
    assert sha(BASE/'plan.json')==frozen['plan_sha256']
    assert sha(BASE/'control_source.tar.gz')==plan['control_archive_sha256']
    with tarfile.open(BASE/'control_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==plan['source_sha256'] and all(sha(ROOT/f)==h for f,h in sources.items())
    assert sha(BASE/'gpu_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'gpu_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    for f in ['tests_v2_job.json','gpu_job.json']:
        job=read(BASE/f);assert job['status']=='finished' and job['exit_code']==0
    assert '16 passed' in (BASE/'tests_focused_v2.log').read_text() and '401 passed' in (BASE/'tests_full_v2.log').read_text()
    assert raw['passed'] and raw['owned_leases_closed'] and len(raw['rows'])==2
    a,b=raw['rows'];assert a['context']==b['context']==4096
    for f in ['state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_h2d_calls','native_d2h_calls','per_layer_initial_d2h_calls']:assert a[f]==b[f],f
    assert len(a['every_visible_state_sha256'])==13
    for x,y in zip(a['trials'],b['trials']):
        for f in ['predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps']:assert x[f]==y[f],f
        assert x['decode_steps']==12 and len(x['predicted_token_ids'])==12
    counts=raw['append_summaries'];assert len(counts)==1
    assert counts[0]==dict(fast_updates=384,fallback_updates=0,pending_released=True,reserved_buffer_allocations=32,shared_hot_prefixes=416,shared_transition_bindings=384,shared_immediate_compactions=0,batched_deferred_compactions=384,bounded_native_groups=48,bounded_scratch_peak_bytes=32473088)
    result=dict(verified=True,tests_passed=401,focused_tests=16,tiny_mistral_contexts_in_reused_append_helper=3,model_prediction_steps=12,cache_visibility_state_frames=13,native_kv_transfers_calls_unchanged=True,append_counts=counts[0],own_hooks_instance_methods_restored=True,
        ready_for_resource_verification=True,ordinary_adoption=False,full_goal_complete=False,
        limits='One4K12 fullmodel pair, tinyCPU model logits exact3contexts. Fullmodel logits not separatelydumped. Serialized standardDynamicLayer instances; existingcustomupdate excluded, concurrentserving notverified. Deferred native groups with immediate cold overflow fallback; scratch bound separately verified. No memory/speed/quality/full512MiB claim.',
        evidence_sha256={f:sha(BASE/f) for f in ['plan.json','gpu_implementation.json','gpu_source.tar.gz','gpu.json','gpu_job.json','gpu.log','tests_job.json','tests_v2_job.json','tests_focused.log','tests_full.log','tests_focused_v2.log','tests_full_v2.log','check_gpu.py','verify_correctness.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
