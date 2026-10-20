"""Frozen3ABBA latency/TTFT regression decision; no metric-driven retry."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    plan=read(BASE/'timing_plan.json');frozen=read(BASE/'timing_implementation.json');raw=read(BASE/'timing.json')
    assert sha(BASE/'timing_plan.json')==frozen['plan_sha256']==raw['plan_sha256']
    assert sha(BASE/'verified_resource.json')==plan['resource_proof_sha256']
    assert sha(BASE/'verified_development.json')==plan['development_proof_sha256']
    assert sha(BASE/'timing_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'timing_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==v for f,v in sources.items())
    job=read(BASE/'timing_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and len(raw['rows'])==12
    observations=raw['quiet_observations'];assert len(observations)==25 and all(r['quiet'] and not r['unexpected_compute_processes'] for r in observations)
    expected=read(BASE/'resource.json')['predicted_token_ids']
    assert len(expected)==50
    summaries={};steps={};ttft={};details=[]
    for block in range(3):
        rows=raw['rows'][block*4:block*4+4]
        assert [r['arm'] for r in rows]==['control','query','query','control']
        assert all(r['block']==block and r['own_hooks_removed'] for r in rows)
        for arm in ('control','query'):
            selected=[r for r in rows if r['arm']==arm]
            values=[];first=[]
            for row in selected:
                assert row['timing_scope']=='end_to_end_model_policy_token'
                assert len(row['trials'])==1
                trial=row['trials'][0];assert trial['decode_steps']==50 and len(trial['step_ms'])==50
                assert trial['predicted_token_ids']==expected
                assert row['selected_layers']==(3200 if arm=='query' else 0)
                assert row['host_read_bytes']>0
                values.extend(trial['step_ms']);first.append(trial['ttft_ms'])
            steps.setdefault(arm,[]).extend(values);ttft.setdefault(arm,[]).extend(first)
            summaries[block,arm]=dict(median_decode_ms=float(np.median(values)),p95_decode_ms=float(np.quantile(values,.95)),
                median_ttft_ms=float(np.median(first)))
        a,b=summaries[block,'control'],summaries[block,'query']
        details.append(dict(block=block,control=a,query=b,
            median_decode_ratio=b['median_decode_ms']/a['median_decode_ms'],p95_decode_ratio=b['p95_decode_ms']/a['p95_decode_ms'],
            ttft_ratio=b['median_ttft_ms']/a['median_ttft_ms']))
    aggregate={arm:dict(median_decode_ms=float(np.median(steps[arm])),p95_decode_ms=float(np.quantile(steps[arm],.95)),
        median_ttft_ms=float(np.median(ttft[arm]))) for arm in ('control','query')}
    a,b=aggregate['control'],aggregate['query']
    ratios=dict(median_decode=b['median_decode_ms']/a['median_decode_ms'],p95_decode=b['p95_decode_ms']/a['p95_decode_ms'],ttft=b['median_ttft_ms']/a['median_ttft_ms'])
    guards=dict(median_decode=ratios['median_decode']<=1.02,p95_decode=ratios['p95_decode']<=1.02,ttft=ratios['ttft']<=1.10)
    passed=all(guards.values())
    result=dict(verified=True,timing_regression_guards_passed=passed,guards=guards,ratios=ratios,aggregate=aggregate,blocks=details,
        measured_decode_forwards_per_arm=300,quiet_compute_process_checks=25,all_teacher_predictions_replayed=True,
        hooks_removed=True,ordinary_adoption=False,full_goal_complete=False,
        wider_candidate_expansion_allowed=passed,
        disposition='Timing pilot passes; independent matched-budget quality remains required before adoption.' if passed else
            'Stop wider candidate quality/timing/adoption: preregistered latency regression guard failed.',
        limits='One model/seed16K50 fixed teacher,3ABBA with one warmup per row. Before/after compute PID checks only, '
            'no continuous GPU utilization assurance. Development quality gain is reused data, not independent NI/superiority. '
            'Full512MiB/exact actual baseline matching/physicalSTT remain open.',
        evidence_sha256={f:sha(BASE/f) for f in ('timing_plan.json','timing_implementation.json','timing_source.tar.gz',
            'timing.json','timing_job.json','verified_resource.json','verified_development.json','verify_timing.py')})
    output=BASE/'verified_timing.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','blocks')},indent=2))
if __name__=='__main__':main()
