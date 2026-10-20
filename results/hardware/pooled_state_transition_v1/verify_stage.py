"""Read-only pooled primitive proof; no model/speed/resource acceptance claim."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def archive(path):
    with tarfile.open(path) as t:return {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
def main():
    output=BASE/'verified_stage.json';assert not output.exists()
    plan=read(BASE/'plan.json');frozen=read(BASE/'gpu_implementation.json');raw=read(BASE/'gpu.json')
    assert sha(BASE/'plan.json')==frozen['plan_sha256']
    assert sha(BASE/'control_source.tar.gz')==plan['control_archive_sha256']
    assert archive(BASE/'control_source.tar.gz')==plan['source_sha256']
    assert all(sha(ROOT/f)==h for f,h in plan['source_sha256'].items())
    assert sha(BASE/'gpu_source.tar.gz')==frozen['archive_sha256'] and archive(BASE/'gpu_source.tar.gz')==frozen['sources']
    assert all(sha(ROOT/f)==h for f,h in frozen['sources'].items())
    for stage in ['tests','gpu']:
        job=read(BASE/(stage+'_job.json'));assert job['status']=='finished' and job['exit_code']==0
    assert '13 passed' in (BASE/'tests_focused.log').read_text() and '450 passed' in (BASE/'tests_full.log').read_text()
    assert raw['passed'] and raw['malformed_native_descriptor_guards']==2 and len(raw['rows'])==5
    assert raw['driver_sha256']==sha(BASE/'check_gpu.py')
    assert sha(Path(raw['binary']))==raw['binary_sha256']
    assert all(r['exact'] for r in raw['rows'])
    prepare=[r for r in raw['rows'] if r['case']=='prepare'];compact=[r for r in raw['rows'] if r['case']=='compact']
    assert [r['layers'] for r in prepare]==[3,32] and [r['layers'] for r in compact]==[1,3,8]
    assert prepare[-1]['owned_bytes']==417792 and prepare[-1]['descriptor_bytes']==384
    assert compact[-1]['scratch_peak_bytes']==32536512 and compact[-1]['descriptor_bytes']==32192
    assert all(r['roots_preserved'] and r['padding_preserved'] for r in compact) and compact[1]['nondefault_stream']
    result=dict(verified=True,tests_passed=450,focused_tests=13,gpu_primitive_cases=5,malformed_native_guards=2,pooled_metadata_owner_bytes=417792,bounded_scratch_peak_bytes=32536512,group8_compaction_descriptor_h2d_bytes=32192,prepare32_descriptor_h2d_bytes=384,bitwise_payload_and_score_age_preservation=True,roots_and_padding_preserved=True,nondefault_stream_with_explicit_dependency_verified=True,ready_for_controller_integration=True,controller_integration_complete=False,ordinary_adoption=False,full_goal_complete=False,
        next='Integrate the shared score/age/timestamp owners and fused prepare/compaction into the complete controller; preserve cold reclaim/fallback ordering. Require fullmodel recorded parity, actual scoped resources and preregistered wholeloop speed gate before wider work.',
        limits=plan['limits']+' No timing comparison was performed. Declared primitive owner/scratch sizes are not fullmodel peak/process memory. Worker stream fixture explicitly waits for source readiness.',
        evidence_sha256={f:sha(BASE/f) for f in ['plan.json','control_source.tar.gz','gpu_implementation.json','gpu_source.tar.gz','tests_focused.log','tests_full.log','tests_job.json','gpu.json','gpu.log','gpu_job.json','check_gpu.py','verify_stage.py']},binary=raw['binary'],binary_sha256=raw['binary_sha256'])
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','binary','binary_sha256']},indent=2))
if __name__=='__main__':main()
