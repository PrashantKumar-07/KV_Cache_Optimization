"""Final read-only ownership resource/timing decision and source/status audit."""
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
    names=['verified_correctness.json','verified_resource.json','verified_cost.json','verified_descriptor_reporting_v2.json'];checked=0
    for name in names:
        proof=read(BASE/name);assert proof['verified'] and not proof['ordinary_adoption'] and not proof['full_goal_complete']
        for f,h in proof['evidence_sha256'].items():assert sha(BASE/f)==h,(name,f);checked+=1
    plan=read(BASE/'plan.json');assert sha(BASE/'control_source.tar.gz')==plan['control_archive_sha256']
    sources=archive(BASE/'control_source.tar.gz');assert sources==plan['source_sha256'] and all(sha(ROOT/f)==h for f,h in sources.items())
    for stage in ['gpu','resource','cost']:
        frozen=read(BASE/(stage+'_implementation.json'));pname='plan.json' if stage=='gpu' else stage+'_plan.json'
        assert sha(BASE/pname)==frozen['plan_sha256']
        assert sha(BASE/(stage+'_source.tar.gz'))==frozen['archive_sha256']
        sources=archive(BASE/(stage+'_source.tar.gz'));assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    raw=read(BASE/'resource.json')
    for field in ['source_sha256','actual_source_sha256']:assert all(sha(ROOT/f)==h for f,h in raw[field].items())
    jobs={p.name:read(p) for p in BASE.glob('*job.json')};assert all(j['status']=='finished' for j in jobs.values())
    assert jobs['tests_job.json']['exit_code']==1
    assert all(j['exit_code']==0 for name,j in jobs.items() if name!='tests_job.json')
    correctness=read(BASE/'verified_correctness.json');assert correctness['tests_passed']==401
    resource=read(BASE/'verified_resource.json');assert resource['resource_primary_passed'] and resource['candidate_active_mib']<=512
    correction=read(BASE/'verified_descriptor_reporting_v2.json');assert correction['candidate_descriptor_h2d_bytes_per_measured_trial_derived']==12864000 and correction['raw_timings_and_failed_decision_unchanged']
    cost=read(BASE/'verified_cost.json');assert not cost['bounded_deferred_wholeloop_primary_passed']
    assert cost['guards']==dict(median_decode=False,saved_ms=False,p95_decode=False,ttft=True)
    manifest=read(BASE/'figure_manifest.json');assert manifest['component_proof_sha256']==sha(BASE/'verified_cost.json') and manifest['resource_proof_sha256']==sha(BASE/'verified_resource.json')
    assert manifest['plot_source_sha256']==sha(BASE/'plot_results.py') and all(sha(BASE/f)==h for f,h in manifest['artifacts_sha256'].items())
    docs=['PROJECT_STATUS.md','START_HERE.md','RESEARCH_ROADMAP.md','PAPER_WORKING_DRAFT.md','REMEDIATION_CHECKLIST.md','SUBMISSION_SPRINT.md']
    for f in docs:
        current=(ROOT/f).read_text()[:3500]
        for term in ['401','432.12','8.15%','63.09','Stop','terminal','withdrawn']:assert term in current,(f,term)
        assert 'is running' not in current and 'timing in progress' not in current
    for f in ['SIMPLE_NOTES.md','explanation.md']:
        current=(ROOT/f).read_text()[:1000]
        for term in ['401','432.12','8.15%','terminal','withdrawn']:assert term in current,(f,term)
    gpurows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    uuid=next(r.split(',')[1].strip() for r in gpurows if r.split(',')[0].strip()=='0')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert not [r for r in apps if r.split(',')[0].strip()==uuid]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),verified=True,classification='Scoped ownership memory passed; whole-loop timing failed. Expansion/adoption stopped.',tests_passed=401,evidence_hash_checks=checked,source_archives_equal_current=True,ordinary_source_cli_defaults_changed=False,candidate_active_mib=resource['candidate_active_mib'],relative_memory_reduction=resource['relative_active_reduction'],median_decode_ratio=cost['ratios']['median_decode'],jobs=jobs,all_owned_jobs_terminal=True,own_hooks_leases_pinned_roots_released=True,gpu0_compute_processes=[],other_gpu_processes_observed_untouched=apps,figure_visually_inspected_no_overlap=True,ordinary_adoption=False,full_goal_complete=False,wider_candidate_expansion=False,unsupported_days_estimate_withdrawn=True,next='Localize added consumed-buffer compaction/transition cost before preregistering any distinct mechanism. Joint latency/memory/independent-quality and physical hardware gates remain open.',evidence_sha256={f:sha(BASE/f) for f in names+['figure_manifest.json','README.md','verify_status.py','failed_implementation.py.txt','failed_tests.py.txt']},status_sha256={f:sha(ROOT/f) for f in docs+['SIMPLE_NOTES.md','explanation.md','EXPERIMENT_LOG.md']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','tests_passed','evidence_hash_checks','candidate_active_mib','median_decode_ratio','all_owned_jobs_terminal','full_goal_complete']},indent=2))
if __name__=='__main__':main()
