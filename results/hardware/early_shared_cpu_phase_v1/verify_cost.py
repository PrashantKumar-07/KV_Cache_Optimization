"""Check frozen trajectories and calculate preregistered timing gates."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    path=BASE/'verified_cost.json';assert not path.exists()
    plan=read(BASE/'cost_plan.json');r=read(BASE/'cost.json');assert r['passed'] and len(r['rows'])==12
    assert read(BASE/'cost_job.json')['exit_code']==0
    assert len(r['quiet_observations'])==25 and all(v['quiet'] for v in r['quiet_observations'])
    for name,value in read(BASE/'cost_helper_sources.json').items():assert sha(ROOT/name)==value
    frozen=read(BASE/'cost_implementation.json');assert sha(BASE/'cost_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'cost_source.tar.gz') as archive:
        for name,value in frozen['sources'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==value
            assert sha(ROOT/name)==value
    assert r['plan_sha256']==sha(BASE/'cost_plan.json')
    summary={}
    for arm in ['snap','cpu']:
        ref=read(ROOT/plan[arm]);assert sha(ROOT/plan[arm])==plan[arm+'_sha256']
        rows=[v for v in r['rows'] if v['arm']==arm];assert len(rows)==6
        for v in rows:
            assert len(v['step_ms'])==50 and v['predicted_token_ids']==ref['predicted_token_ids']
            if arm=='cpu':
                assert v['state_sha256']==ref['state_sha256'] and v['counters']==ref['counters'] and v['early_shared_layers']==32
                for field,key in [('h2d_payload_bytes','host_read_bytes'),('d2h_payload_bytes','host_write_bytes'),('native_h2d_calls','native_h2d_calls'),('native_d2h_calls','native_d2h_calls')]:assert v[field]==ref['prototype_counts'][key]
            else:assert v['state_sha256']==ref['final_cache_sha256']
        steps=[t for v in rows for t in v['step_ms']]
        summary[arm]=dict(median_ms=float(np.median(steps)),p95_ms=float(np.percentile(steps,95)),ttft_ms=float(np.median([v['ttft_ms'] for v in rows])),prefill_peak_bytes=max(v['prefill_peak_gpu_bytes'] for v in rows),measured_forwards=len(steps))
    a=summary['snap'];b=summary['cpu'];gain=100*(1-b['median_ms']/a['median_ms']);saved=a['median_ms']-b['median_ms'];p95=100*(b['p95_ms']/a['p95_ms']-1);ttft=100*(b['ttft_ms']/a['ttft_ms']-1);prefill=100*(b['prefill_peak_bytes']/a['prefill_peak_bytes']-1)
    gates=dict(median_gain=gain>=5,median_saved=saved>=4,p95=p95<=2,ttft=ttft<=10,prefill=prefill<=10)
    result=dict(verified=True,summary=summary,median_gain_pct=gain,median_saved_ms=saved,p95_change_pct=p95,ttft_change_pct=ttft,prefill_change_pct=prefill,gates=gates,joint_timing_gate_passed=all(gates.values()),full_trajectory_resources_equal=False,ordinary_adoption=False,full_goal_complete=False,all_owned_jobs_terminal=True,disposition='Proceed only if all preregistered gates pass; failed primary/guard stops candidate expansion/adoption. Initial resource equality does not prove fulltrajectory equality.',evidence_sha256={name:sha(BASE/name) for name in ['cost_plan.json','cost.json','cost.log','cost_job.json','cost_implementation.json','cost_source.tar.gz','cost_helper_sources.json','verified_resources.json']},verifier_sha256=sha(Path(__file__)))
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
