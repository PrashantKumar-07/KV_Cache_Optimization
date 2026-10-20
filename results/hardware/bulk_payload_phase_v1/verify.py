"""Verify frozen primitive parity/cost gates and build identity."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    out=BASE/'verified_stage.json';assert not out.exists();gpu=read(BASE/'gpu.json');cost=read(BASE/'cost.json')
    assert gpu['passed'] and len(gpu['checks'])==4 and len(gpu['rejections'])==6
    assert all(v['exact'] for v in gpu['checks']) and all(v['rejected_before_mutation'] for v in gpu['rejections'])
    assert any(v['case']=='exhaustive_new_BF16' for v in gpu['checks']) and any(v['case']=='empty_hot' for v in gpu['rejections'])
    assert sha(Path(gpu['binary_path']))==gpu['binary_sha256']
    for kind in ['gpu','cost']:
        frozen=read(BASE/(kind+'_implementation.json'));assert sha(BASE/(kind+'_source.tar.gz'))==frozen['archive_sha256']
        with tarfile.open(BASE/(kind+'_source.tar.gz')) as t:
            for name,value in frozen['sources'].items():
                assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==value,name
                assert sha(ROOT/name)==value,name
        assert read(BASE/(kind+'_job.json'))['exit_code']==0
    assert read(BASE/'tests_job.json')['exit_code']==0 and read(BASE/'tests_job.json')['suite_passed']==503
    assert cost['passed'] and cost['advance_to_model'] and len(cost['rows'])==36
    assert cost['plan_sha256']==sha(BASE/'cost_plan.json')
    summary=[]
    for hot,all_paid in [(649,False),(991,False),(649,True)]:
        rows=[v for v in cost['rows'] if v['hot']==hot and v['all_paid']==all_paid];assert len(rows)==12
        byarm={arm:[t for v in rows if v['arm']==arm for t in v['step_ms']] for arm in ['current','bulk']}
        assert all(len(v)==120 for v in byarm.values())
        median={k:float(np.median(v)) for k,v in byarm.items()};p95={k:float(np.percentile(v,95)) for k,v in byarm.items()}
        saved=median['current']-median['bulk'];assert saved>=4 and p95['bulk']<=1.02*p95['current']
        expected=next(v for v in cost['summary'] if (v['hot'],v['all_paid'])==(hot,all_paid));assert expected['exact'] and expected['saved_ms']==saved and expected['median_ms']==median
        summary.append(expected)
    result=dict(verified=True,tests_passed=503,bit_parity_passed=True,malformed_batch_atomic_rejection_passed=True,charged_stage_gate_passed=True,advance_to_model=True,summary=summary,model_integrated=False,model_latency_measured=False,complete512MiB_measured=False,quality_measured=False,ordinary_adoption=False,full_goal_complete=False,all_owned_jobs_terminal=True,binary_sha256=gpu['binary_sha256'],limits=cost['limits'],next='Wire standalone group8 executor into opt-in CPUphase lifecycle, preserving CPUowner/versions, slotFIFO, metrics, native KV submitted bytes/calls and leases. Prove real-model parity/resources before timing; ordinary defaults unchanged.',evidence_sha256={p.name:sha(p) for p in BASE.iterdir() if p.suffix in ['.json','.log','.gz']},verifier_sha256=sha(Path(__file__)))
    out.write_text(json.dumps(result,indent=2)+'\n');print('Verified:503tests,4bitparitycases,6atomicrejections,threechargedgatespass;modelintegrationpending.')
if __name__=='__main__':main()
