"""Final read-only audit: failed CPUtimestamp primary, no adoption."""
import hashlib,json,subprocess,tarfile
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def archive(path):
    with tarfile.open(path) as t:return {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
def main():
    output=BASE/'verified_status.json';assert not output.exists()
    proof_names=['verified_correctness.json','verified_resource.json','verified_cost.json'];checked=0
    for name in proof_names:
        proof=read(BASE/name);assert proof['verified'] and not proof['ordinary_adoption'] and not proof['full_goal_complete']
        for f,h in proof['evidence_sha256'].items():assert sha(BASE/f)==h,(name,f);checked+=1
    plan=read(BASE/'plan.json');assert sha(BASE/'control_source.tar.gz')==plan['control_archive_sha256']
    source=archive(BASE/'control_source.tar.gz');assert source==plan['source_sha256'] and all(sha(ROOT/f)==h for f,h in source.items())
    for stage in ['gpu','resource','cost']:
        frozen=read(BASE/(stage+'_implementation.json'));plan_name='plan.json' if stage=='gpu' else stage+'_plan.json'
        assert sha(BASE/plan_name)==frozen['plan_sha256']
        assert sha(BASE/(stage+'_source.tar.gz'))==frozen['archive_sha256']
        source=archive(BASE/(stage+'_source.tar.gz'));assert source==frozen['sources'] and all(sha(ROOT/f)==h for f,h in source.items())
    resource_raw=read(BASE/'resource.json')
    for field in ['source_sha256','actual_source_sha256']:
        assert all(sha(ROOT/f)==h for f,h in resource_raw[field].items())
    jobs={stage:read(BASE/(stage+'_job.json')) for stage in ['tests','gpu','resource','cost']}
    assert all(j['status']=='finished' and j['exit_code']==0 for j in jobs.values())
    correctness=read(BASE/'verified_correctness.json');assert correctness['tests_passed']==358
    resource=read(BASE/'verified_resource.json');assert resource['resource_primary_passed'] and not resource['full512mib_passed']
    cost=read(BASE/'verified_cost.json');assert not cost['cpu_timestamp_wholeloop_primary_passed']
    assert cost['guards']==dict(median_decode=False,saved_ms=False,p95_decode=True,ttft=True)
    manifest=read(BASE/'figure_manifest.json');assert manifest['component_proof_sha256']==sha(BASE/'verified_cost.json') and manifest['resource_proof_sha256']==sha(BASE/'verified_resource.json')
    assert manifest['plot_source_sha256']==sha(BASE/'plot_results.py') and all(sha(BASE/f)==h for f,h in manifest['artifacts_sha256'].items())
    docs=['PROJECT_STATUS.md','START_HERE.md','RESEARCH_ROADMAP.md','PAPER_WORKING_DRAFT.md','REMEDIATION_CHECKLIST.md','SUBMISSION_SPRINT.md']
    for f in docs:
        current=(ROOT/f).read_text()[:3000]
        for term in ['358','541.19','0.88%','0.51ms','Stop standalone','terminal','withdrawn']:assert term in current,(f,term)
        assert 'is running' not in current and 'timing in progress' not in current
    for f in ['SIMPLE_NOTES.md','explanation.md']:
        current=(ROOT/f).read_text()[:1000]
        for term in ['358','0.88%','541.19','withdrawn']:assert term in current,(f,term)
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    uuid=next(r.split(',')[1].strip() for r in gpus if r.split(',')[0].strip()=='0')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert not [r for r in apps if r.split(',')[0].strip()==uuid]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),verified=True,classification='Progress: CPUtimestamp mechanism preserves ordinarymodel state/choices and charges metadata; completewholeloop primaryfailed, expansionstopped.',
        tests_passed=358,evidence_hash_checks=checked,source_archives_equal_current=True,ordinary_source_cli_defaults_changed=False,
        candidate_active_envelope_plus_CPU_tensor_bound_mib=resource['candidate_active_plus_cpu_metadata_tensor_bound_mib'],component_median_gain_percent=100*cost['relative_median_improvement'],component_saved_ms=cost['median_decode_saved_ms'],
        jobs=jobs,all_owned_jobs_terminal=True,own_hooks_mirrors_pinned_roots_released=True,gpu0_compute_processes=[],other_gpu_processes_observed_untouched=apps,
        figure_visually_inspected_no_overlap=True,ordinary_adoption=False,full_goal_complete=False,wider_candidate_expansion=False,unsupported_days_estimate_withdrawn=True,
        next='Preregister broader hot/cold state-transition and ownership restructuring; projection-only and timestamp-only hypotheses have failed. Require complete charged/wholeloop/resource and eventualfaithfulbaseline/independentquality gates.',
        evidence_sha256={f:sha(BASE/f) for f in proof_names+['figure_manifest.json','README.md','verify_status.py']},status_sha256={f:sha(ROOT/f) for f in docs+['SIMPLE_NOTES.md','explanation.md','EXPERIMENT_LOG.md']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','tests_passed','evidence_hash_checks','component_median_gain_percent','component_saved_ms','all_owned_jobs_terminal','full_goal_complete']},indent=2))
if __name__=='__main__':main()
