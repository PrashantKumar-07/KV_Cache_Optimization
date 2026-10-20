"""Verify preserved runs and localize changed calls; no speed acceptance claim."""
import hashlib,json,pstats,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def entry(stats,suffix,name):
    matches=[(k,v) for k,v in stats.stats.items() if k[0].endswith(suffix) and k[2]==name]
    assert len(matches)==1,(suffix,name,matches)
    k,(primitive,total,selftime,inclusive,callers)=matches[0]
    return dict(file=k[0],line=k[1],function=name,calls=total,self_seconds=selftime,inclusive_seconds=inclusive,inclusive_ms_per_decode=inclusive/50*1000)
def main():
    output=BASE/'verified_diagnostic.json';assert not output.exists()
    plan=read(BASE/'plan.json');frozen=read(BASE/'source.json')
    assert sha(BASE/'plan.json')==frozen['plan_sha256'] and sha(BASE/'source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    assert sha(ROOT/plan['failed_decision'])==plan['failed_decision_sha256']
    assert not read(ROOT/plan['failed_decision'])['shared_transition_wholeloop_primary_passed']
    job=read(BASE/'profile_job.json');assert job['status']=='finished' and job['exit_code']==0
    summaries={}
    for arm in ['control','candidate']:
        raw=read(BASE/(arm+'.json'));refpath=ROOT/plan[arm]
        assert sha(refpath)==plan[arm+'_sha256'];ref=read(refpath)
        assert raw['passed'] and raw['plan_sha256']==frozen['plan_sha256']
        assert raw['frozen50predictions_state_counters_payload_calls_replayed']
        assert raw['state_sha256']==ref['state_sha256'] and raw['counters']==ref['counters']
        assert raw['profiled_timing_not_latency_gate']['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        assert len(ref['predicted_token_ids'])==50
        assert raw['hooks_restored'] and raw['owned_host_after_cleanup_bytes']==0
        assert len(raw['quiet_observations'])==2 and all(r['quiet'] for r in raw['quiet_observations'])
        if arm=='candidate':assert raw['lease_closed']
        for f,v in raw['payload'].items():assert v==ref['prototype_counts'][f]
        for f in ['native_h2d_calls','native_d2h_calls']:assert raw[f]==ref['prototype_counts'][f]
        stats=pstats.Stats(str(BASE/(arm+'.prof')))
        summary=dict(total_calls=stats.total_calls,profiled_policy_seconds=stats.total_tt,
            eviction=entry(stats,'<deferred_production_eviction>','evict_and_demote'),
            preparation=entry(stats,'batched_eviction_policy.py','_prepare_decode_reads'),
            visibility=entry(stats,'reuse_visible_policy.py','_visible_tuple'))
        if arm=='control':
            summary.update(compaction=entry(stats,'fused_eviction_policy.py','_compact_or_queue'),native_gather=entry(stats,'native_reuse_visible.py','gather_into'))
            assert summary['native_gather']['calls']==50
        else:
            summary.update(compaction=entry(stats,'shared_decode_transition.py','_compact_or_queue'),in_place_gather=entry(stats,'raw_input_reuse.py','gather_in_place'),visible_reservation=entry(stats,'reserved_decode_append.py','reserve_visible'),hot_extension=entry(stats,'raw_input_reuse.py','_extend_hot_pair'))
            assert summary['in_place_gather']['calls']==1600
        assert summary['compaction']['calls']==1600
        summaries[arm]=summary
    result=dict(verified=True,scope=plan['scope'],summaries=summaries,owned_jobs_terminal=True,source_archives_equal_current=True,all_frozen_pred_state_counters_transfers_replayed=True,
        diagnosis='The measured shared design substitutes 1600 per-layer in-place gathers for 50 native batched gathers, with added reservation/extension checks. This explains changed work and localizes regression; profiling cannot establish exclusive causation or acceptance latency.',
        code_trace='After demotion, cold reclamation reads compacted positions/cumulative scores. It does not read compacted hot KV. Payload can be deferred with bounded alias-safe scratch; score compaction must precede reclaim on overflow.',
        next='Do not rerun the failed implementation. A distinct batched ownership transition would need scoped memory and ordinary whole-loop gates. The remaining controller preparation/visibility/eviction costs require broader consolidation to close the faithful-baseline latency gap.',
        ordinary_adoption=False,full_goal_complete=False,failed_timing_decision_preserved=True,limits=plan['limits'],evidence_sha256={f:sha(BASE/f) for f in ['plan.json','source.json','source.tar.gz','profile_pair.py','profile_job.json','profile.log','control.json','candidate.json','control.prof','candidate.prof','control.prof.txt','candidate.prof.txt','verify_diagnostic.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','summaries','diagnosis','full_goal_complete']},indent=2))
if __name__=='__main__':main()
