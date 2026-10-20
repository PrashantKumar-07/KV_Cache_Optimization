"""Frozen3ABBA ordinaryLRU guard for GPU-source-map full-query candidate."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    plan=read(BASE/'lru_plan.json');frozen=read(BASE/'lru_implementation.json');raw=read(BASE/'lru.json')
    assert sha(BASE/'lru_plan.json')==frozen['plan_sha256']==raw['plan_sha256']
    assert sha(BASE/'verified_cost.json')==plan['component_proof_sha256']
    assert sha(BASE/'verified_resource_v2.json')==plan['resource_proof_sha256']
    assert sha(ROOT/plan['query_reference'])==plan['query_reference_sha256']
    assert sha(ROOT/plan['control'])==plan['control_sha256']
    assert sha(BASE/'lru_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'lru_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==v for f,v in sources.items())
    job=read(BASE/'lru_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and len(raw['rows'])==12
    observations=raw['quiet_observations'];assert len(observations)==25 and all(r['quiet'] and not r['unexpected_compute_processes'] for r in observations)
    references=dict(control=read(ROOT/plan['control']),query=read(ROOT/plan['query_reference']));expected=references['control']['predicted_token_ids']
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
                assert row['visible_suffix_updates']==(3200 if arm=='query' else 0)
                assert row['native_h2d_calls']==(1701 if arm=='query' else 76)
                assert row['native_d2h_calls']==(78 if arm=='query' else 57)
                reference=references[arm]
                assert row['frozen_arm_state_transfers_exact'] and row['state_sha256']==reference['state_sha256']
                assert row['logical_counters']==reference['counters']
                assert row['host_read_bytes']==reference['prototype_counts']['host_read_bytes']
                assert row['host_write_bytes']==reference['prototype_counts']['host_write_bytes']
                if arm=='query':
                    assert row['index_metadata_h2d_bytes']==75333120 and row['index_refreshes']==51 and row['index_reads']==1600
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
    result=dict(verified=True,ordinary_lru_regression_guards_passed=passed,guards=guards,ratios=ratios,aggregate=aggregate,blocks=details,
        measured_decode_forwards_per_arm=300,quiet_compute_process_checks=25,all_teacher_predictions_replayed=True,
        hooks_removed=True,ordinary_adoption=False,full_goal_complete=False,
        candidate_index_metadata_h2d_bytes_per_measured_trial=75333120,unchanged_native_h2d_kv_bytes_per_trial=5825961984,
        disposition='OrdinaryLRU latency guards pass; independent matched-baseline quality/full512MiB still required before adoption.' if passed else
            'Stop standalone suffix fullquery quality/timing expansion/adoption: ordinaryLRU latency guard failed. Verified7.85% composition gain remains research-only.',
        limits=plan['limits']+' Before/after GPU0 compute PID checks only, no continuous utilization assurance.',
        relative_median_change=ratios['median_decode']-1,
        median_decode_added_ms=b['median_decode_ms']-a['median_decode_ms'],
        fullscan_quality_timing_expansion=False,
        evidence_sha256={f:sha(BASE/f) for f in ('lru_plan.json','lru_implementation.json','lru_source.tar.gz',
            'lru.json','lru_job.json','verified_cost.json','verified_resource_v2.json','verify_lru.py')})
    output=BASE/'verified_lru.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','blocks')},indent=2))
if __name__=='__main__':main()
