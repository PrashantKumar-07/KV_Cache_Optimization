"""Read-only repair of profiler annotation scope, preserving every failed run."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda path:json.loads(path.read_text())


def main():
    output=BASE/'verified_summary.json';assert not output.exists()
    plan=read(BASE/'plan.json');raw=read(BASE/'profile_v4.json')
    frozen=read(BASE/'implementation_v4.json');correction=read(BASE/'annotation_scope_correction_v2.json')
    assert raw['passed'] is False and raw['error']=='AssertionError: '
    assert raw['fixed_query_state_counters_transfers_replayed']
    assert raw['measured_selected_layers']==1600
    assert raw['phase_ast']['annotations_unwrap_to_exact_candidate']
    assert raw['plan_sha256']==frozen['plan_sha256']==sha(BASE/'plan.json')
    assert frozen['archive_sha256']==sha(BASE/'implementation_source_v4.tar.gz')
    assert frozen['correction_sha256']==sha(BASE/'report_retention_correction.json')
    with tarfile.open(BASE/'implementation_source_v4.tar.gz') as archive:
        sources={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                 for m in archive.getmembers() if m.isfile()}
    assert sources==frozen['sources']
    assert all(sha(ROOT/name)==expected for name,expected in sources.items())
    assert all(sha(ROOT/name)==expected for name,expected in plan['source_sha256'].items())
    assert correction['supersedes_sha256']==sha(BASE/'annotation_scope_correction.json')
    assert correction['raw_report_sha256']==sha(BASE/'profile_v4.json')
    assert correction['raw_job_sha256']==sha(BASE/'profile_v4_job.json')
    assert correction['original_driver_sha256']==sha(BASE/'check_profile_v4.py')
    assert all(sha(Path(name))==expected for name,expected in correction['local_primary_source_sha256'].items())
    # Check the executed reporting flag occurs strictly after every preservation
    # assertion and before the erroneous name-only aggregation assertion.
    driver=(BASE/'check_profile_v4.py').read_text()
    flag=driver.index('fixed_query_state_counters_transfers_replayed=True')
    for text in ["result['trials'][0]['predicted_token_ids'] == reference['predicted_token_ids']",
                 "digest(policy.layer_caches) == reference['state_sha256']",
                 "policy._live_totals() == reference['counters']",
                 "== reference['prototype_counts'][key]",
                 'policy.manager.h2d_calls == 1701 and policy.manager.d2h_calls == 78',
                 'policy.query_reader.metadata_h2d_bytes == 75333120']:
        assert driver.index(text)<flag
    assert flag<driver.index("assert len(phases) == 8 and all(row['count']==1600")
    for key in ('control','prompt','teachers'):
        assert sha(ROOT/plan[key])==plan[key+'_sha256']
    reference=read(ROOT/plan['control']);assert len(reference['predicted_token_ids'])==50
    assert reference['prototype_counts']['host_read_bytes']==5825961984
    assert reference['prototype_counts']['host_write_bytes']==115855360
    observations=raw['quiet_observations']
    assert len(observations)==5 and all(r['quiet'] and not r['unexpected_compute_processes'] for r in observations)
    assert raw['own_hooks_removed'] and raw['index_metadata_released']
    assert '287 passed' in (BASE/'tests_full.log').read_text()
    assert '4 passed' in (BASE/'tests_focused_v2.log').read_text()
    jobs={name:read(BASE/(name+'_job.json')) for name in ['profile','profile_v3','profile_v4','tests']}
    assert all(job['status']=='finished' for job in jobs.values())
    assert all(jobs[name]['exit_code']==1 for name in ['profile','profile_v3','profile_v4'])
    assert jobs['tests']['exit_code']==0
    cpu=[row for row in raw['phases'] if row['cpu_elapsed_total_ms']>0]
    spans=[row for row in raw['phases'] if row['cpu_elapsed_total_ms']==0]
    assert len(cpu)==8 and len(spans)==6
    assert all(row['count']==1600 for row in cpu+spans)
    assert {row['name'].split('/')[1] for row in cpu}==set(plan['phases'])
    assert all(row['cpu_self_total_ms']==0 and row['device_self_total_ms']==row['attributed_device_total_ms']>0 for row in spans)
    total=sum(row['cpu_elapsed_total_ms'] for row in cpu)
    rows=[dict(phase=row['name'].split('/')[1],observations=row['count'],
        instrumented_cpu_elapsed_ms_per_decode=row['cpu_elapsed_total_ms']/50,
        instrumented_cpu_self_ms_per_decode=row['cpu_self_total_ms']/50,
        share_of_cpu_phase_ranges=row['cpu_elapsed_total_ms']/total,
        incomplete_torch_correlated_gpu_ms_per_decode=row['attributed_device_total_ms']/50) for row in cpu]
    csv_path=BASE/'phase_costs.csv';assert not csv_path.exists()
    with csv_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    largest=max(rows,key=lambda row:row['instrumented_cpu_elapsed_ms_per_decode'])
    assert largest['phase']=='gpu_visible_composition'
    assert not (BASE/'trace_v4.json').exists()
    names=['plan.json','implementation.json','implementation_v2.json','implementation_v3.json','implementation_v4.json',
           'implementation_source_v4.tar.gz','profile.log','profile_v3.log','profile_v4.log',
           'profile_v3.json','profile_v4.json','profile_job.json','profile_v3_job.json','profile_v4_job.json',
           'tests_job.json','tests_full.log','tests_focused.log','tests_focused_v2.log','isolation_correction.json',
           'report_retention_correction.json','annotation_scope_correction.json','annotation_scope_correction_v2.json','verify_summary.py','phase_costs.csv']
    result=dict(verified=True,classification='Verified scoped CPU-phase diagnostic summary; original trace jobs failed reporting guards.',
        tests_passed=287,focused_tests_passed=4,expected_cpu_phases=8,observations_per_phase=1600,
        duplicate_zero_cpu_gpu_annotation_spans=6,phase_costs=rows,largest_cpu_elapsed_phase=largest,
        exact_50_step_frozen_query_state_counters_transfer_assertions_executed=True,
        fidelity_evidence='Executed assertions and flag in immutable v4 driver/report, linked to frozen native-query reference; not a separately dumped new trajectory.',
        native_kv_h2d_bytes=5825961984,native_kv_d2h_bytes=115855360,candidate_index_metadata_h2d_bytes=75333120,
        ordinary_source_cli_defaults_unchanged=True,quiet_compute_process_checks=5,
        own_hooks_and_index_metadata_released=True,all_owned_jobs_terminal=True,
        original_failed_reports_and_counts_preserved=True,chrome_trace_exported=False,
        native_kernel_gpu_attribution_complete=False,speed_or_adoption_proof=False,
        next=correction['next'],ordinary_adoption=False,fullscan_quality_timing_expansion=False,full_goal_complete=False,
        limits=plan['limits']+' Chrome trace not exported; custom-native-kernel attribution incomplete. GPU annotation spans are not kernel sums.',
        evidence_sha256={name:sha(BASE/name) for name in names})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified_summary':True,'largest_cpu_phase':largest,
        'chrome_trace_exported':False,'full_goal_complete':False},indent=2))

if __name__=='__main__':main()
