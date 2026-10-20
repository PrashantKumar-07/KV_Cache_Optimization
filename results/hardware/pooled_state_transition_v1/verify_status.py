"""Final read-only stage lineage, process cleanup and status consistency audit."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_status.json';assert not output.exists()
    proof=read(BASE/'verified_stage.json');assert proof['verified'] and proof['ready_for_controller_integration'] and not proof['controller_integration_complete'] and not proof['full_goal_complete']
    assert all(sha(BASE/f)==h for f,h in proof['evidence_sha256'].items())
    assert sha(Path(proof['binary']))==proof['binary_sha256']
    frozen=read(BASE/'gpu_implementation.json');assert all(sha(ROOT/f)==h for f,h in frozen['sources'].items())
    jobs={stage:read(BASE/(stage+'_job.json')) for stage in ['tests','gpu']};assert all(j['status']=='finished' and j['exit_code']==0 for j in jobs.values())
    docs=['PROJECT_STATUS.md','START_HERE.md','RESEARCH_ROADMAP.md','PAPER_WORKING_DRAFT.md','REMEDIATION_CHECKLIST.md','SUBMISSION_SPRINT.md']
    for name in docs:
        text=(ROOT/name).read_text()[:2200]
        for word in ['450','417792','31.03','integration remains incomplete','terminal','withdrawn']:assert word in text,(name,word)
        assert 'is running' not in text
    for name in ['SIMPLE_NOTES.md','explanation.md']:
        text=(ROOT/name).read_text()[:800]
        for word in ['450','417792','31.03','unproven','terminal','withdrawn']:assert word in text,(name,word)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    uuid=next(r.split(',')[1].strip() for r in rows if r.split(',')[0].strip()=='0')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert not [r for r in apps if r.split(',')[0].strip()==uuid]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),verified=True,classification='Progress: pooled GPU transition primitives preserve tested bits/roots/padding/stream order; full controller integration remains incomplete.',tests_passed=450,gpu_primitive_cases=5,jobs=jobs,all_owned_jobs_terminal=True,source_archives_equal_current=True,gpu0_compute_processes=[],other_gpu_processes_observed_untouched=apps,ordinary_source_cli_defaults_changed=False,controller_integration_complete=False,ordinary_adoption=False,full_goal_complete=False,unsupported_day_estimate_withdrawn=True,next=proof['next'],evidence_sha256={f:sha(BASE/f) for f in ['verified_stage.json','README.md','verify_status.py']},status_sha256={f:sha(ROOT/f) for f in docs+['SIMPLE_NOTES.md','explanation.md','EXPERIMENT_LOG.md']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','tests_passed','gpu_primitive_cases','all_owned_jobs_terminal','controller_integration_complete','full_goal_complete']}))
if __name__=='__main__':main()
