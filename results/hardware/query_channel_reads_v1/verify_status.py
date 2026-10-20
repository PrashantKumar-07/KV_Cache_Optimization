"""Final evidence/source/job/status audit; failed primary remains stopped."""
import hashlib,json,subprocess,tarfile
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_status.json';assert not output.exists()
    names=['verified_correctness.json','verified_resource.json','verified_cost.json'];checked=0
    for name in names:
        proof=read(BASE/name);assert proof['verified'] and not proof['ordinary_adoption'] and not proof['full_goal_complete']
        for f,h in proof['evidence_sha256'].items():assert sha(BASE/f)==h,(name,f);checked+=1
    archive_pairs=[('implementation.json','implementation_source.tar.gz','plan.json'),
        ('gpu_implementation.json','gpu_source.tar.gz','plan.json'),
        ('resource_implementation.json','resource_source.tar.gz','resource_plan.json'),
        ('cost_implementation.json','cost_source.tar.gz','cost_plan.json')]
    for metadata_name,archive_name,plan_name in archive_pairs:
        frozen=read(BASE/metadata_name)
        assert sha(BASE/plan_name)==frozen['plan_sha256']
        assert sha(BASE/archive_name)==frozen['archive_sha256']
        with tarfile.open(BASE/archive_name) as t:
            source={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
        assert source==frozen['sources']
        assert all(sha(ROOT/f)==h for f,h in source.items())
    plan=read(BASE/'plan.json');assert all(sha(ROOT/f)==h for f,h in plan['source_sha256'].items())
    native=read(BASE/'native_v2.json');assert sha(Path(native['binary_path']))==native['binary_sha256']
    jobs={name:read(BASE/(name+'_job.json')) for name in ['tests','native','native_v2','gpu','resource','cost']}
    assert all(j['status']=='finished' for j in jobs.values())
    assert jobs['native']['exit_code']==1 and all(j['exit_code']==0 for n,j in jobs.items() if n!='native')
    correctness=read(BASE/'verified_correctness.json');assert correctness['tests_passed']==328
    resource=read(BASE/'verified_resource.json');assert resource['resource_primary_passed'] and not resource['full512mib_passed']
    cost=read(BASE/'verified_cost.json');assert not cost['channel_component_primary_passed']
    assert cost['guards']==dict(median_decode=False,saved_ms=True,p95_decode=True,ttft=True)
    assert not cost['fullscan_quality_timing_expansion']
    assert not (BASE/'lru.json').exists(),'Failed component must not proceed to ordinaryLRU gate'
    manifest=read(BASE/'figure_manifest.json')
    assert manifest['resource_proof_sha256']==sha(BASE/'verified_resource.json')
    assert manifest['component_proof_sha256']==sha(BASE/'verified_cost.json') and manifest['plot_source_sha256']==sha(BASE/'plot_results.py')
    assert all(sha(BASE/f)==h for f,h in manifest['artifacts_sha256'].items())
    docs=['PROJECT_STATUS.md','START_HERE.md','RESEARCH_ROADMAP.md','PAPER_WORKING_DRAFT.md','REMEDIATION_CHECKLIST.md','SUBMISSION_SPRINT.md']
    for f in docs:
        current=(ROOT/f).read_text()[:3000]
        for term in ['328','45.53%','545.17','4.57%','Stop standalone','10–20','terminal']:
            assert term in current,(f,term)
        assert 'is running' not in current
    for f in ['SIMPLE_NOTES.md','explanation.md']:
        current=(ROOT/f).read_text()[:1600]
        for term in ['328','45.53%','4.57%','terminal','10–20']:assert term in current,(f,term)
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    uuid=next(line.split(',')[1].strip() for line in lines if line.split(',')[0].strip()=='0')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    gpu0=[line for line in apps if line.split(',')[0].strip()==uuid];assert not gpu0
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),verified=True,
        classification='Progress: exact key/selected-value reader reduces copiedKV payload; component primary failed, standalone expansion stopped.',
        tests_passed=328,evidence_hash_checks=checked,source_archives_equal_current=True,native_binary_exact=True,
        ordinary_source_cli_defaults_unchanged=True,jobs=jobs,all_owned_jobs_terminal=True,
        gpu0_compute_processes=gpu0,gpu_snapshot=lines,other_gpu_processes_observed_untouched=[line for line in apps if line not in gpu0],
        figure_visually_inspected_no_overlap=True,resource_active_mib=resource['candidate_active_decode_envelope_mib'],
        copied_kv_payload_reduction_percent=100*resource['copied_kv_payload_reduction'],component_median_gain_percent=100*cost['relative_median_improvement'],
        component_primary_passed=False,ordinary_lru_gate_run=False,ordinary_adoption=False,fullscan_quality_timing_expansion=False,full_goal_complete=False,
        next='Preregister a distinct query-loop/synchronization mechanism; prove correctness/resources/component/LRU before expansion. No favorable retry of the failed channel gate.',
        remaining_working_days_estimate=[10,20],estimate_condition='Conditional on hardware and research outcomes.',
        evidence_sha256={f:sha(BASE/f) for f in names+['figure_manifest.json','README.md','verify_status.py']},
        status_sha256={f:sha(ROOT/f) for f in docs+['SIMPLE_NOTES.md','explanation.md','EXPERIMENT_LOG.md']})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['verified','tests_passed','evidence_hash_checks','copied_kv_payload_reduction_percent','component_median_gain_percent','all_owned_jobs_terminal','full_goal_complete']},indent=2))
if __name__=='__main__':main()
