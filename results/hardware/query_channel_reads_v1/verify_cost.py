"""Read-only frozen channel component gate; copied bytes differ by design."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_cost.json';assert not output.exists()
    plan=read(BASE/'cost_plan.json');frozen=read(BASE/'cost_implementation.json');raw=read(BASE/'cost.json')
    assert sha(BASE/'cost_plan.json')==frozen['plan_sha256']==raw['plan_sha256']
    for name,key in [('verified_correctness.json','correctness_proof_sha256'),('verified_resource.json','resource_proof_sha256')]:
        assert sha(BASE/name)==plan[key]
    assert read(BASE/'verified_resource.json')['ready_for_component_cost_gate']
    assert sha(BASE/'cost_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'cost_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==v for f,v in sources.items())
    assert read(BASE/'cost_job.json')['status']=='finished' and read(BASE/'cost_job.json')['exit_code']==0
    assert raw['passed'] and len(raw['rows'])==12
    assert len(raw['quiet_observations'])==25 and all(r['quiet'] and not r['unexpected_compute_processes'] for r in raw['quiet_observations'])
    native=read(BASE/'native_v2.json');assert sha(Path(native['binary_path']))==native['binary_sha256']
    refs={arm:read(ROOT/plan[arm]) for arm in ('control','query')}
    assert all(sha(ROOT/plan[arm])==plan[arm+'_sha256'] for arm in refs)
    steps={};ttft={};details=[]
    for block in range(3):
        rows=raw['rows'][block*4:block*4+4];assert [r['arm'] for r in rows]==['control','query','query','control']
        summaries={}
        for arm in ('control','query'):
            values=[];first=[];ref=refs[arm]
            for row in [r for r in rows if r['arm']==arm]:
                assert row['block']==block and row['own_hooks_removed']
                assert row['timing_scope']=='end_to_end_model_policy_token' and len(row['trials'])==1
                trial=row['trials'][0];assert trial['decode_steps']==50 and len(trial['step_ms'])==50
                assert trial['predicted_token_ids']==ref['predicted_token_ids'] and len(ref['predicted_token_ids'])==50
                assert row['selected_layers']==row['visible_suffix_updates']==3200
                assert row['frozen_query_state_transfers_exact'] and row['state_sha256']==ref['state_sha256']
                assert row['logical_counters']==ref['counters']
                for field in ('host_read_bytes','host_write_bytes'):assert row[field]==ref['prototype_counts'][field]
                for field in ('native_h2d_calls','native_d2h_calls'):assert row[field]==ref[field]
                assert row['index_metadata_h2d_bytes']==75333120 and row['index_refreshes']==51
                assert row['index_reads']==(3200 if arm=='query' else 1600)
                assert row['native_channel_binary_sha256']==native['binary_sha256']
                expected={f:ref['channel_'+f] for f in ['key_reads','value_reads','key_rows','value_rows']} if arm=='query' else {}
                assert row['channel_counts']==expected
                values.extend(trial['step_ms']);first.append(trial['ttft_ms'])
            steps.setdefault(arm,[]).extend(values);ttft.setdefault(arm,[]).extend(first)
            summaries[arm]=dict(median_decode_ms=float(np.median(values)),p95_decode_ms=float(np.quantile(values,.95)),median_ttft_ms=float(np.median(first)))
        a,b=summaries['control'],summaries['query']
        details.append(dict(block=block,control=a,query=b,median_decode_ratio=b['median_decode_ms']/a['median_decode_ms'],p95_decode_ratio=b['p95_decode_ms']/a['p95_decode_ms'],ttft_ratio=b['median_ttft_ms']/a['median_ttft_ms']))
    aggregate={arm:dict(median_decode_ms=float(np.median(steps[arm])),p95_decode_ms=float(np.quantile(steps[arm],.95)),median_ttft_ms=float(np.median(ttft[arm]))) for arm in steps}
    a,b=aggregate['control'],aggregate['query'];ratios={k:b[v]/a[v] for k,v in [('median_decode','median_decode_ms'),('p95_decode','p95_decode_ms'),('ttft','median_ttft_ms')]}
    saved=a['median_decode_ms']-b['median_decode_ms']
    guards=dict(median_decode=ratios['median_decode']<=.95,saved_ms=saved>=4,p95_decode=ratios['p95_decode']<=1.02,ttft=ratios['ttft']<=1.10)
    passed=all(guards.values())
    result=dict(verified=True,channel_component_primary_passed=passed,guards=guards,ratios=ratios,aggregate=aggregate,blocks=details,relative_median_improvement=1-ratios['median_decode'],median_decode_saved_ms=saved,
        measured_decode_forwards_per_arm=300,quiet_compute_process_checks=25,all_teacher_predictions_replayed=True,hooks_removed=True,
        control_copied_kv_h2d_bytes=refs['control']['prototype_counts']['host_read_bytes'],candidate_copied_kv_h2d_bytes=refs['query']['prototype_counts']['host_read_bytes'],
        ordinary_adoption=False,full_goal_complete=False,fullscan_quality_timing_expansion=False,
        disposition='Component passed; preregister fresh ordinaryLRU guard before wider retrieval/adoption.' if passed else 'Stop standalone channel-reader expansion/adoption: component primary failed.',limits=plan['limits']+' Before/after PID observations, not continuous utilization.',
        evidence_sha256={f:sha(BASE/f) for f in ['cost_plan.json','cost_implementation.json','cost_source.tar.gz','cost.json','cost_job.json','verified_correctness.json','verified_resource.json','native_v2.json','verify_cost.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','blocks')},indent=2))
if __name__=='__main__':main()
