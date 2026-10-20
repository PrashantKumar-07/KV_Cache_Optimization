"""Verify isolated diagnostic trace, fixed native-query replay and archives."""
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
    output=BASE/'verified_profile.json';assert not output.exists()
    plan=read(BASE/'plan.json');raw=read(BASE/'profile_v3.json')
    frozen=read(BASE/'implementation_v3.json');correction=read(BASE/'isolation_correction.json')
    assert raw['passed'] and raw['frozen_query_replay']
    assert frozen['plan_sha256']==raw['plan_sha256']==sha(BASE/'plan.json')
    assert frozen['archive_sha256']==sha(BASE/'implementation_source_v3.tar.gz')
    assert frozen['correction_sha256']==sha(BASE/'isolation_correction.json')
    with tarfile.open(BASE/'implementation_source_v3.tar.gz') as archive:
        sources={member.name:hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                 for member in archive.getmembers() if member.isfile()}
    assert sources==frozen['sources']
    assert all(sha(ROOT/name)==expected for name,expected in sources.items())
    assert all(sha(ROOT/name)==expected for name,expected in plan['source_sha256'].items())
    assert correction['failed_job_sha256']==sha(BASE/'profile_job.json')
    assert correction['failed_log_sha256']==sha(BASE/'profile.log')
    assert correction['original_driver_sha256']==sha(BASE/'check_profile.py')
    assert correction['tests_full_sha256']==sha(BASE/'tests_full.log')
    assert read(BASE/'profile_job.json')['exit_code']==1
    assert read(BASE/'profile_v3_job.json')['status']=='finished'
    assert read(BASE/'profile_v3_job.json')['exit_code']==0
    assert read(BASE/'tests_job.json')['exit_code']==0
    assert '287 passed' in (BASE/'tests_full.log').read_text()
    assert '4 passed' in (BASE/'tests_focused_v2.log').read_text()
    for key in ('control','prompt','teachers'):
        assert sha(ROOT/plan[key])==plan[key+'_sha256']
    reference=read(ROOT/plan['control'])
    assert raw['predicted_token_ids']==reference['predicted_token_ids']
    assert raw['state_sha256']==reference['state_sha256']
    assert raw['logical_counters']==reference['counters']
    assert raw['native_h2d_bytes']==reference['prototype_counts']['host_read_bytes']==5825961984
    assert raw['native_d2h_bytes']==reference['prototype_counts']['host_write_bytes']==115855360
    assert raw['native_h2d_calls']==1701 and raw['native_d2h_calls']==78
    assert raw['index_metadata_h2d_bytes']==75333120 and raw['selected_layers']==1600
    assert raw['own_hooks_removed'] and raw['index_metadata_released']
    assert len(raw['quiet_observations'])==5
    assert all(item['quiet'] and not item['unexpected_compute_processes'] for item in raw['quiet_observations'])
    assert raw['phase_ast']['annotations_unwrap_to_exact_candidate']
    stop=read(ROOT/'results/hardware/gpu_query_read_indices_v1/verified_lru.json')
    assert stop['ordinary_lru_regression_guards_passed'] is False
    assert sha(ROOT/'results/hardware/gpu_query_read_indices_v1/verified_lru.json')==plan['prior_stop_proof_sha256']
    assert sha(BASE/'trace_v3.json')==raw['trace_sha256']
    assert raw['profiler_cpu_events']>0 and raw['profiler_cuda_events']>0
    phases=raw['phases']
    assert len(phases)==8 and all(row['count']==1600 for row in phases)
    assert {row['name'].split('/')[1] for row in phases}==set(plan['phases'])
    assert all(row['cpu_elapsed_total_ms']>=row['cpu_self_total_ms']>=0 and
               row['attributed_device_total_ms']>=0 for row in phases)
    assert sum(row['attributed_device_total_ms'] for row in phases)>0
    rows=[]
    for row in phases:
        rows.append(dict(phase=row['name'].split('/')[1],observations=row['count'],
            cpu_elapsed_ms_per_decode=row['cpu_elapsed_total_ms']/50,
            cpu_self_ms_per_decode=row['cpu_self_total_ms']/50,
            attributed_gpu_kernel_ms_per_decode=row['attributed_device_total_ms']/50))
    csv_path=BASE/'phase_costs.csv';assert not csv_path.exists()
    with csv_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    largest_cpu=max(rows,key=lambda row:row['cpu_elapsed_ms_per_decode'])
    largest_gpu=max(rows,key=lambda row:row['attributed_gpu_kernel_ms_per_decode'])
    result=dict(verified=True,tests_passed=287,focused_tests_passed=4,
        exact_prediction_steps=50,selected_layer_forwards=1600,
        fixed_query_final_state_counters_transfer_bytes_and_calls_replayed=True,
        ordinary_source_cli_defaults_unchanged=True,scoped_annotations_unwrap_to_exact_gpu_index_candidate=True,
        native_h2d_bytes=raw['native_h2d_bytes'],native_d2h_bytes=raw['native_d2h_bytes'],
        candidate_index_metadata_h2d_bytes=raw['index_metadata_h2d_bytes'],
        phase_costs=rows,largest_cpu_elapsed_phase=largest_cpu,largest_attributed_gpu_phase=largest_gpu,
        quiet_process_checks=5,own_hooks_and_index_metadata_released=True,
        initial_concurrent_test_attempt_aborted_before_warmup=True,failed_attempt_preserved=True,
        isolated_diagnostic_job_finished=True,ordinary_adoption=False,full_goal_complete=False,
        fullscan_quality_timing_expansion=False,limits=plan['limits'],
        evidence_sha256={name:sha(BASE/name) for name in ['plan.json','implementation.json',
            'implementation_v2.json','implementation_v3.json','implementation_source_v3.tar.gz',
            'profile_v3.json','profile_v3_job.json','profile_v3.log','isolation_correction.json',
            'profile_job.json','profile.log','tests_full.log','tests_focused.log','tests_focused_v2.log',
            'verify_profile.py','phase_costs.csv']},trace_sha256=raw['trace_sha256'])
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified':True,'phase_costs':rows,'full_goal_complete':False},indent=2))

if __name__=='__main__':main()
