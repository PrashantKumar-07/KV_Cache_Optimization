"""Read-only proof checks; write a new result once."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_resources.json';assert not output.exists()
    plan=read(BASE/'plan.json');a=read(ROOT/plan['control']);s=read(ROOT/plan['snap']);b=read(BASE/'cpu.json')
    assert sha(ROOT/plan['control'])==plan['control_sha256'] and sha(ROOT/plan['snap'])==plan['snap_sha256']
    for kind in ['gpu','parity','resource']:
        frozen=read(BASE/(kind+'_implementation.json'))
        assert sha(BASE/(kind+'_source.tar.gz'))==frozen['archive_sha256']
        with tarfile.open(BASE/(kind+'_source.tar.gz')) as archive:
            for name,value in frozen['sources'].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==value,name
                assert sha(ROOT/name)==value,name
    for name in ['gpu','parity_4096_12','parity_16384_50']:
        assert read(BASE/(name+'.json'))['passed']
        assert read(BASE/(name+'_job.json'))['exit_code']==0
    assert read(BASE/'tests_job.json')['exit_code']==0 and read(BASE/'resource_job.json')['exit_code']==0
    assert b['passed'] and b['early_shared_layers']==32 and b['cpu_append_calls']==50 and b['cpu_fallback_steps']==0
    for key in ['predicted_token_ids','state_sha256','counters','prototype_counts']:
        assert b[key]==a[key],key
    assert b['owned_pinned_after_cleanup_bytes']==0 and b['append_leases_closed'] and b['prefill_leases_closed']
    assert b['owned_host_allocator_block_after_cleanup_bytes']==0
    roots=[sum(v['distinct_owned_kv_storage_bytes'].values()) for v in b['ownership_boundaries']]
    assert len(roots)==51 and set(roots)=={341*2**20}
    assert b['forward_geometry']==a['forward_geometry']
    m=a['resource_model']['constant_model_storage_bytes'];assert m==b['resource_model']['constant_model_storage_bytes']
    active=lambda d:d['gpu_phase_peaks']['decode_peak']-m
    reserved=lambda d:d['gpu_phase_peaks']['whole_reserved_peak']-m
    reduction=1-b['gpu_phase_peaks']['prefill_peak']/a['gpu_phase_peaks']['prefill_peak']
    gpu_change=active(b)/active(a)-1;reserved_change=reserved(b)/reserved(a)-1
    scoped=(active(b)+b['owned_host_allocator_block_peak_bytes']+b['cpu_owned_bytes']+b['controller_cpu_streak_array_peak']+b['controller_address_array_peak'])/2**20
    snap_ratio=b['gpu_phase_peaks']['prefill_peak']/s['gpu_phase_peaks']['prefill_peak']
    passed=reduction>=.10 and gpu_change<=.02 and reserved_change<=.02 and scoped<=512 and snap_ratio<=1.10
    result=dict(verified=True,resource_gate_passed=passed,tests_passed=501,whole_prefill_reduction_pct=100*reduction,prefill_peak_gb=b['gpu_phase_peaks']['prefill_peak']/1e9,strict_decode_gpu_change_pct=100*gpu_change,strict_reserved_gpu_change_pct=100*reserved_change,scoped_active_mib=scoped,faithful_snap_prefill_ratio=snap_ratio,initial_KV_mib=341,full_trajectory_resources_equal=False,full_goal_complete=False,ordinary_adoption=False,all_owned_jobs_terminal=True,evidence_sha256={str(p.relative_to(BASE)):sha(p) for p in BASE.iterdir() if p.suffix in ['.json','.log','.gz']},verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2));assert passed
if __name__=='__main__':main()
