"""Verify native frozen replay and suffix ownership without rerunning models."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def main():
    output=BASE/'verified_correctness.json';assert not output.exists()
    plan=read(BASE/'plan.json');frozen=read(BASE/'implementation.json');raw=read(BASE/'gpu.json')
    assert sha(BASE/'plan.json')==frozen['plan_sha256']
    assert sha(BASE/'implementation_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'implementation_source.tar.gz') as archive:
        sources={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                 for m in archive.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==v for f,v in sources.items())
    assert all(sha(ROOT/f)==v for f,v in plan['source_sha256'].items())
    for k in ('control','resource_control','diagnostic'):
        assert sha(ROOT/plan[k])==plan[k+'_sha256']
    assert '19 passed' in (BASE/'tests_focused.log').read_text()
    assert '306 passed' in (BASE/'tests_full.log').read_text()
    assert read(BASE/'tests_job.json')['exit_code']==0
    job=read(BASE/'gpu_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and raw['hooks_canonical_method_restored']
    assert raw['suffix_ast']['only_composition_block_changed']
    assert raw['suffix_ast']['control']==raw['read_ast']['candidate']
    assert len(raw['arms'])==2
    reference=read(ROOT/plan['control'])['arms'][1]
    for arm in raw['arms']:
        assert arm['canonical_reads_during_selector']==0
        assert arm['prefix_canonical_ownership_checks']==384
        assert len(arm['predicted_token_ids'])==12 and len(arm['frames'])==13 and len(arm['selections'])==384
        for field in ['predicted_token_ids','frames','selections','host_read_bytes','host_write_bytes',
            'host_initial_write_bytes','query_index_metadata_h2d_bytes','query_index_refreshes','query_index_reads']:
            assert arm[field]==reference[field]
    candidate=raw['arms'][1]
    assert candidate['retained_visible_destinations']==candidate['visible_suffix_updates']==384
    replay=read(BASE/'frozen_replay.json');assert replay['verified'] and replay['raw_sha256']==sha(BASE/'gpu.json')
    assert replay['frozen_gpu_index_arm_sha256']==plan['control_sha256']
    result=dict(verified=True,tests_passed=306,focused_tests=19,
        exact_prediction_steps=12,exact_logit_cache_counter_frames=13,exact_visible_kv_choice_observations=384,
        prefix_and_canonical_hot_kv_ownership_checks_per_arm=384,retained_candidate_visible_destinations=384,
        host_kv_h2d_bytes=candidate['host_read_bytes'],host_kv_d2h_bytes=candidate['host_write_bytes'],
        candidate_index_metadata_h2d_bytes=candidate['query_index_metadata_h2d_bytes'],
        query_native_reads=candidate['query_index_reads'],canonical_cpu_reads=0,
        frozen_gpu_index_query_replay=True,only_composition_block_changed=True,
        ownhooks_and_index_cleanup=True,ordinary_source_cli_defaults_unchanged=True,
        ready_for_fresh_resource_gate=True,ordinary_adoption=False,full_goal_complete=False,
        limits='One model/seed4K12 correctness, CPU bitwise suffix/offset/stride/alias cases. Prefix is guaranteed by unchanged ordinary visible preparation and checked before/after every native hook. No speed/independentquality/full512MiB/physicalSTT/adoption claim.',
        evidence_sha256={f:sha(BASE/f) for f in ['plan.json','control_source.tar.gz','implementation.json',
            'implementation_source.tar.gz','tests_focused.log','tests_full.log','tests_job.json','gpu_job.json',
            'gpu.json','gpu.log','frozen_replay.json','verify_correctness.py']})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
