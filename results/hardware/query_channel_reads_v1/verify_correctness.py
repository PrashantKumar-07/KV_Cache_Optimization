"""Verify immutable native checks, unchanged query states and channel-byte ledger."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    output=BASE/'verified_correctness.json';assert not output.exists()
    plan=read(BASE/'plan.json');metadata=read(BASE/'gpu_implementation.json')
    assert metadata['plan_sha256']==sha(BASE/'plan.json')
    assert metadata['archive_sha256']==sha(BASE/'gpu_source.tar.gz')
    with tarfile.open(BASE/'gpu_source.tar.gz') as archive:
        sources={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
    assert sources==metadata['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    assert all(sha(ROOT/f)==h for f,h in plan['source_sha256'].items())
    build=read(BASE/'native_build_plan_v2.json')
    assert metadata['native_build_plan_sha256']==sha(BASE/'native_build_plan_v2.json')
    for name,key in [('implementation_source.tar.gz','original_source_archive_sha256'),('check_native.py','original_driver_sha256'),
        ('native.json','original_report_sha256'),('native.log','original_log_sha256'),('native_job.json','original_job_sha256'),
        ('check_native_v2.py','new_driver_sha256')]:assert sha(BASE/name)==build[key]
    assert read(BASE/'native_job.json')['exit_code']==1
    assert read(BASE/'native_v2_job.json')['exit_code']==0
    assert read(BASE/'gpu_job.json')['exit_code']==0
    assert read(BASE/'tests_job.json')['exit_code']==0
    assert '22 passed' in (BASE/'tests_focused.log').read_text() and '328 passed' in (BASE/'tests_full.log').read_text()
    native=read(BASE/'native_v2.json');assert native['passed'] and native['roots_and_metadata_released']
    assert len(native['cases'])==3
    for case in native['cases']:
        assert len(case['metadata_guards'])==12
        for key in ['exact_channel_bits','accounted_channel_bytes','stale_prefetch_guards','serial_stream_handoff',
                    'alternating_pool_local_mapping','empty_reads_no_dispatch']:assert case[key]
    assert sha(Path(native['binary_path']))==native['binary_sha256']
    assert all(sha(ROOT/'experiments'/name)==h for name,h in native['native_source_sha256'].items())
    raw=read(BASE/'gpu.json');assert raw['passed'] and raw['hooks_canonical_method_restored']
    control,candidate=raw['arms'];ref=read(ROOT/plan['control'])['arms'][1]
    assert sha(ROOT/plan['control'])==plan['control_sha256']
    for arm in [control,candidate]:
        assert len(arm['predicted_token_ids'])==12 and len(arm['frames'])==13 and len(arm['selections'])==384
        assert arm['prefix_canonical_ownership_checks']==arm['retained_visible_destinations']==arm['visible_suffix_updates']==384
        assert arm['canonical_reads_during_selector']==0
        for key in ['predicted_token_ids','frames','selections','host_write_bytes','host_initial_write_bytes',
                    'query_index_metadata_h2d_bytes','query_index_refreshes']:assert arm[key]==ref[key]
    ledger=read(BASE/'transfer_accounting.json');assert ledger['verified'] and ledger['raw_sha256']==sha(BASE/'gpu.json')
    assert ledger['channel_ast']['ranking_math_unchanged'] and ledger['native_binary_sha256']==native['binary_sha256']
    assert control['host_read_bytes']==ref['host_read_bytes']==ledger['control_total_kv_h2d_bytes']
    assert candidate['host_read_bytes']==ledger['candidate_total_kv_h2d_bytes']
    assert ledger['key_rows']==candidate['scanned_rows']==control['scanned_rows']
    assert ledger['value_rows']==384*32
    assert candidate['channel_key_reads']==candidate['channel_value_reads']==384 and candidate['query_index_reads']==768
    assert ledger['candidate_query_h2d_bytes']==(ledger['key_rows']+ledger['value_rows'])*8*128*2
    assert ledger['control_query_h2d_bytes']==2*ledger['key_rows']*8*128*2
    assert candidate['host_read_bytes']==ledger['common_nonquery_kv_h2d_bytes']+ledger['candidate_query_h2d_bytes']
    assert control['host_read_bytes']==ledger['common_nonquery_kv_h2d_bytes']+ledger['control_query_h2d_bytes']
    assert candidate['native_h2d_calls']==control['native_h2d_calls']+384
    assert candidate['native_d2h_calls']==control['native_d2h_calls']
    scope=read(BASE/'gpu_stdout_scope_correction.json')
    assert scope['raw_log_sha256']==sha(BASE/'gpu.log') and scope['raw_report_sha256']==sha(BASE/'gpu.json')
    assert scope['accounting_sha256']==sha(BASE/'transfer_accounting.json')
    result=dict(verified=True,tests_passed=328,focused_tests=22,native_dtypes=3,native_metadata_guards=36,
        exact_prediction_steps=12,exact_logit_cache_counter_frames=13,exact_visible_kv_choice_observations=384,
        retained_visible_destinations_and_prefix_checks_per_arm=384,ranking_math_unchanged=True,
        control_copied_kv_h2d_bytes=control['host_read_bytes'],candidate_copied_kv_h2d_bytes=candidate['host_read_bytes'],
        copied_kv_payload_reduction=1-candidate['host_read_bytes']/control['host_read_bytes'],
        unchanged_d2h_bytes=candidate['host_write_bytes'],unchanged_batch_index_metadata_h2d_bytes=candidate['query_index_metadata_h2d_bytes'],
        native_binary_sha256=native['binary_sha256'],native_cpp_sources_sha256=native['native_source_sha256'],
        canonical_cpu_reads=0,ownhooks_roots_and_index_cleanup=True,ordinary_source_cli_defaults_unchanged=True,
        ready_for_fresh_resource_gate=True,ordinary_adoption=False,full_goal_complete=False,
        limits='One model/seed4K12 exactness. Copied channelKV payload counted separately from batchindexmetadata; not all library H2D/PCIe/physicalSTT traffic. Numerical indices trusted from validated maps/topk; no parallel serving/independentquality/latency/full512MiB/adoption claim.',
        evidence_sha256={name:sha(BASE/name) for name in ['plan.json','gpu_implementation.json','gpu_source.tar.gz',
            'native_build_plan_v2.json','native.json','native.log','native_job.json','native_v2.json','native_v2.log','native_v2_job.json',
            'tests_focused.log','tests_full.log','tests_job.json','gpu.json','gpu.log','gpu_job.json','frozen_replay.json',
            'transfer_accounting.json','gpu_stdout_scope_correction.json','verify_correctness.py']})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
