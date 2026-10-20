"""Frozen development pilot direction and current-API control replay."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
FIELDS=('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
        'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes')


def main():
    output=BASE/'verified_development.json';assert not output.exists()
    plan=read(BASE/'development_plan.json');m=read(BASE/'development_implementation.json');raw=read(BASE/'development.json')
    assert sha(BASE/'development_plan.json')==m['plan_sha256']
    assert sha(BASE/'development_source.tar.gz')==m['archive_sha256']
    with tarfile.open(BASE/'development_source.tar.gz') as t:
        sources={f.name:hashlib.sha256(t.extractfile(f).read()).hexdigest() for f in t.getmembers() if f.isfile()}
    assert sources==m['sources'] and all(sha(ROOT/f)==v for f,v in sources.items())
    assert sha(BASE/'verified_resource.json')==plan['resource_proof_sha256']
    assert sha(ROOT/plan['development_manifest'])==plan['development_manifest_sha256']
    assert sha(ROOT/plan['frozen_control_rows'])==plan['frozen_control_rows_sha256']
    job=read(BASE/'development_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and len(raw['rows'])==24 and raw['holdout_context_overlap']==0
    assert raw['development_plan_sha256']==m['plan_sha256']
    old=read(ROOT/plan['frozen_control_rows']);oldrows={}
    for row in old['rows']:oldrows.setdefault((row['task'],row['sample_id']),[]).append(row)
    pairs={}
    for row in raw['rows']:
        key=row['task'],row['sample_id'];assert row['arm'] not in pairs.setdefault(key,{})
        pairs[key][row['arm']]=row
        assert row['reserved_host_tensor_bytes']==(1090+64)*131072
        assert row['selector_hooks_removed']
        assert row['observed']['query_enabled']==(row['arm']=='candidate')
        if row['arm']=='candidate':assert row['observed']['selected_layers']>0
    assert len(pairs)==12
    per_task={};changes=selected_layers=scanned_rows=0;details=[]
    for (task,sample),arms in pairs.items():
        assert set(arms)=={'control','candidate'}
        a,b=arms['control'],arms['candidate']
        assert a['prompt_sha256']==b['prompt_sha256'] and a['generation']['prompt_tokens']==b['generation']['prompt_tokens']
        assert len(oldrows[task,sample])==2
        assert any(all(a[f]==frozen[f] for f in FIELDS) for frozen in oldrows[task,sample]),('Control replay failed',task,sample)
        delta=b['score']-a['score'];per_task.setdefault(task,[]).append(delta)
        changed=a['generated_token_ids']!=b['generated_token_ids'];changes+=changed
        selected_layers+=b['observed']['selected_layers'];scanned_rows+=b['observed']['scanned_rows']
        details.append(dict(task=task,sample_id=sample,control_score=a['score'],candidate_score=b['score'],
            score_difference=delta,prediction_ids_changed=changed,
            control_h2d_bytes=a['explicit_h2d_payload_bytes'],candidate_h2d_bytes=b['explicit_h2d_payload_bytes']))
    assert len(per_task)==6 and all(len(v)==2 for v in per_task.values())
    direction=float(np.mean([np.mean(v) for v in per_task.values()]))
    # Descriptive uncertainty for development only, not the final independent quality gate.
    rng=np.random.default_rng(20261003);draws=np.stack([
        np.asarray(v)[rng.integers(0,2,size=(10000,2))].mean(1) for v in per_task.values()]).mean(0)
    ci=list(map(float,np.quantile(draws,[.025,.975])))
    pilot_passed=direction>=0
    expand=pilot_passed
    result=dict(verified=True,development_pairs=12,tasks=6,frozen_control_api_replay=True,
        task_balanced_score_difference=direction,descriptive_development_bootstrap_ci95=ci,
        bootstrap_seed=20261003,bootstrap_draws=10000,pilot_nonnegative_direction_passed=pilot_passed,
        changed_prediction_pairs=changes,selected_layer_forwards=selected_layers,scanned_cold_rows=scanned_rows,
        quality_improvement_proved=False,
        ready_for_preregistered_timing_guard=expand,ordinary_integration_ready=False,
        disposition='Development direction permits quiet3ABBA timing guard; independent quality remains open.' if expand else
            'Stop wider quality/timing/adoption: negative development direction.',
        per_task={task:dict(pairs=2,mean_score_difference=float(np.mean(v))) for task,v in per_task.items()},
        pairs=details,full_goal_complete=False,ordinary_source_changed=False,all_owned_jobs_finished=True,
        limits='Two reused development requests/task and one seed; descriptive bootstrap only, no independent NI/superiority. '
            'Full-scan current-query native H2D payloads charged; observer wall times invalid for latency claims. '
            'Exact actual baseline allocation/width match, '
            'complete512MiB, timing and physicalSTT remain open.',
        evidence_sha256={f:sha(BASE/f) for f in ('development_plan.json','development_implementation.json','development_source.tar.gz',
            'verified_resource.json','development.json','development_job.json')},verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','pairs')},indent=2))


if __name__=='__main__':main()
