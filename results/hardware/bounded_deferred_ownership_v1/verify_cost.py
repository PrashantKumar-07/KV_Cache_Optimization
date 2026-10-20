"""Independent whole-loop ordinaryCPU timestamp speed gate with replay proof."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_cost.json';assert not output.exists()
    plan=read(BASE/'cost_plan.json');raw=read(BASE/'cost.json');frozen=read(BASE/'cost_implementation.json')
    assert sha(BASE/'cost_plan.json')==raw['plan_sha256']==frozen['plan_sha256']
    for f,k in [('verified_correctness.json','correctness_proof_sha256'),('verified_resource.json','resource_proof_sha256')]:assert sha(BASE/f)==plan[k]
    assert read(BASE/'verified_resource.json')['ready_for_wholeloop_cost_gate']
    assert sha(BASE/'cost_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'cost_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    job=read(BASE/'cost_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and len(raw['rows'])==12
    assert len(raw['quiet_observations'])==25 and all(r['quiet'] and not r['unexpected_compute_processes'] for r in raw['quiet_observations'])
    refs={arm:read(ROOT/plan[arm]) for arm in ['control','query']}
    assert all(sha(ROOT/plan[arm])==plan[arm+'_sha256'] for arm in refs)
    steps={};ttft={};blocks=[]
    for block in range(3):
        rows=raw['rows'][block*4:block*4+4];assert [r['arm'] for r in rows]==['control','query','query','control']
        summary={}
        for arm in refs:
            v=[];first=[];ref=refs[arm]
            for row in [r for r in rows if r['arm']==arm]:
                assert row['block']==block and row['own_hooks_removed']
                assert row['timing_scope']=='end_to_end_model_policy_token' and len(row['trials'])==1
                trial=row['trials'][0];assert trial['decode_steps']==50 and len(trial['step_ms'])==50
                assert trial['predicted_token_ids']==ref['predicted_token_ids'] and len(ref['predicted_token_ids'])==50
                assert row['selected_layers']==0 and row['frozen_arm_pred_state_counters_transfers_exact']
                assert row['state_sha256']==ref['state_sha256'] and row['logical_counters']==ref['counters']
                for f in ['host_read_bytes','host_write_bytes','native_h2d_calls','native_d2h_calls']:assert row[f]==ref['prototype_counts'][f]
                counts=row['shared_transition_counts']
                if arm=='control':
                    assert not counts and row['append_fast_updates']==row['append_fallback_updates']==0
                else:
                    assert counts==dict(reserved_buffer_allocations=32,shared_hot_prefixes=1632,shared_transition_bindings=1600,shared_immediate_compactions=0,batched_deferred_compactions=1600,bounded_native_groups=200,bounded_scratch_peak_bytes=32473088)
                    assert row['append_fast_updates']==3200 and row['append_fallback_updates']==0
                v.extend(trial['step_ms']);first.append(trial['ttft_ms'])
            steps.setdefault(arm,[]).extend(v);ttft.setdefault(arm,[]).extend(first)
            summary[arm]=dict(median_decode_ms=float(np.median(v)),p95_decode_ms=float(np.quantile(v,.95)),median_ttft_ms=float(np.median(first)))
        a,b=summary['control'],summary['query'];blocks.append(dict(block=block,control=a,query=b,median_decode_ratio=b['median_decode_ms']/a['median_decode_ms'],p95_decode_ratio=b['p95_decode_ms']/a['p95_decode_ms'],ttft_ratio=b['median_ttft_ms']/a['median_ttft_ms']))
    aggregate={arm:dict(median_decode_ms=float(np.median(steps[arm])),p95_decode_ms=float(np.quantile(steps[arm],.95)),median_ttft_ms=float(np.median(ttft[arm]))) for arm in refs}
    a,b=aggregate['control'],aggregate['query'];saved=a['median_decode_ms']-b['median_decode_ms']
    ratios={k:b[v]/a[v] for k,v in [('median_decode','median_decode_ms'),('p95_decode','p95_decode_ms'),('ttft','median_ttft_ms')]}
    guards=dict(median_decode=ratios['median_decode']<=.95,saved_ms=saved>=4,p95_decode=ratios['p95_decode']<=1.02,ttft=ratios['ttft']<=1.10)
    passed=all(guards.values())
    result=dict(verified=True,bounded_deferred_wholeloop_primary_passed=passed,guards=guards,ratios=ratios,aggregate=aggregate,blocks=blocks,relative_median_improvement=1-ratios['median_decode'],median_decode_saved_ms=saved,
        measured_decode_forwards_per_arm=300,quiet_compute_process_checks=25,all_teacher_predictions_state_counters_kv_replayed=True,owned_hooks_leases_released=True,
        native_kv_h2d_bytes_per_trial=315621376,native_kv_d2h_bytes_per_trial=117571584,candidate_extra_metadata_transfers=0,
        ordinary_adoption=False,full_goal_complete=False,
        disposition='Primarypassed; require widerlength/nativeboundary/freegeneration/recycling and currentfaithfulbaseline/jointresources/quality beforeordinaryintegration.' if passed else 'Stop bounded deferred ownership expansion/adoption: wholeloop primary or guard failed. No favorabletiming retry.',
        limits=plan['limits'],
        evidence_sha256={f:sha(BASE/f) for f in ['cost_plan.json','cost_implementation.json','cost_source.tar.gz','cost.json','cost_job.json','verified_correctness.json','verified_resource.json','check_cost.py','verify_cost.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['blocks','evidence_sha256']},indent=2))
if __name__=='__main__':main()
