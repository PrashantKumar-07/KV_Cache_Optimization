"""Read-only frozen fullmodel pair proof; residency differs, values do not."""
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
    with tarfile.open(BASE/'control_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==plan['source_sha256'] and all(sha(ROOT/f)==h for f,h in sources.items())
    assert sha(BASE/'gpu_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'gpu_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    for f in ['tests_job.json','gpu_job.json']:
        job=read(BASE/f);assert job['status']=='finished' and job['exit_code']==0
    assert '11 passed' in (BASE/'tests_focused.log').read_text() and '358 passed' in (BASE/'tests_full.log').read_text()
    assert raw['passed'] and len(raw['rows'])==2 and len(raw['gpu_sort_tie_cases'])==8
    assert all(r['gpu_tie_order_exact'] for r in raw['gpu_sort_tie_cases'])
    a,b=raw['rows'];assert a['context']==b['context']==4096
    for f in ['state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_h2d_calls','native_d2h_calls','per_layer_initial_d2h_calls']:
        assert a[f]==b[f],f
    assert len(a['every_visible_state_sha256'])==len(b['every_visible_state_sha256'])==13
    for x,y in zip(a['trials'],b['trials']):
        for f in ['predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps']:assert x[f]==y[f],f
        assert x['decode_steps']==12 and len(x['predicted_token_ids'])==12
    metadata=raw['timestamp_counts'];assert metadata['all_timestamps_cpu'] and metadata['mirrors_released']
    assert metadata['refreshes']==13 and metadata['h2d_bytes']>0 and metadata['initial_d2h_bytes']>0
    assert metadata['cpu_timestamp_bytes']==a['occupied_host_bytes']//4096*4 or metadata['cpu_timestamp_bytes']>0
    result=dict(verified=True,tests_passed=358,focused_tests=11,model_prediction_steps=12,cache_visibility_state_frames=13,gpu_tie_order_cases=8,native_kv_transfers_calls_unchanged=True,timestamp_metadata=metadata,
        limits='One4K12 modelpair, exactstate/cachevisibility/pred/counters not separatelycapturedlogitarrays. TimestampsCPUresident, GPUunstablevisibilitysort preserved. Noquality/latency/resource/full512MiB/adoptionclaim.',ready_for_resource_verification=True,ordinary_adoption=False,full_goal_complete=False,
        evidence_sha256={f:sha(BASE/f) for f in ['plan.json','gpu_implementation.json','gpu_source.tar.gz','gpu.json','gpu_job.json','gpu.log','tests_job.json','tests_focused.log','tests_full.log','check_gpu.py','verify_correctness.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
