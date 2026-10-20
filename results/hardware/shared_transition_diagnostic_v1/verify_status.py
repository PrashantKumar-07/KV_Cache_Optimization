"""Read-only diagnostic lineage, cleanup and current status consistency."""
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_status.json';assert not output.exists()
    proof=read(BASE/'verified_diagnostic.json');assert proof['verified'] and not proof['full_goal_complete']
    for f,h in proof['evidence_sha256'].items():assert sha(BASE/f)==h
    source=read(BASE/'source.json');assert all(sha(ROOT/f)==h for f,h in source['sources'].items())
    previous=ROOT/'results/hardware/shared_decode_transition_v1'
    for name in ['verified_correctness.json','verified_resource.json','verified_cost.json']:
        prior=read(previous/name);assert prior['verified']
        assert all(sha(previous/f)==h for f,h in prior['evidence_sha256'].items())
    docs=['PROJECT_STATUS.md','START_HERE.md','RESEARCH_ROADMAP.md','PAPER_WORKING_DRAFT.md','REMEDIATION_CHECKLIST.md','SUBMISSION_SPRINT.md']
    for f in docs:
        text=(ROOT/f).read_text()[:1800]
        assert all(word in text for word in ['1600','50 batched','consolidation','terminal','incomplete','withdrawn']),f
    for f in ['SIMPLE_NOTES.md','explanation.md']:
        text=(ROOT/f).read_text()[:700];assert all(word in text for word in ['1600','50 batched','incomplete','withdrawn']),f
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    uuid=next(r.split(',')[1].strip() for r in rows if r.split(',')[0].strip()=='0')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert not [r for r in apps if r.split(',')[0].strip()==uuid]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),verified=True,classification='Progress: regression localized with fresh paired preserved-behavior profiles; prior failed timing remains final.',source_lineage_current=True,all_owned_jobs_terminal=True,gpu0_compute_processes=[],other_gpu_processes_observed_untouched=apps,ordinary_adoption=False,full_goal_complete=False,day_estimate_withdrawn=True,evidence_sha256={f:sha(BASE/f) for f in ['verified_diagnostic.json','README.md','verify_status.py']},status_sha256={f:sha(ROOT/f) for f in docs+['SIMPLE_NOTES.md','explanation.md','EXPERIMENT_LOG.md']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','all_owned_jobs_terminal','full_goal_complete']}))
if __name__=='__main__':main()
