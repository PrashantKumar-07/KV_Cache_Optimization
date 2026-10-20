"""Final read-only proof/source/job/status audit for diagnostic and corpus."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
CORPUS=ROOT/'results/hardware/controller_transition_corpus_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=BASE/'verified_status.json';assert not out.exists()
for base,name in [(BASE,'verified_diagnostic.json'),(CORPUS,'verified_corpus.json')]:
 r=read(base/name);assert r['verified'] and not r['full_goal_complete']
 for f,h in r['evidence_sha256'].items():assert sha(base/f)==h,(name,f)
 sources=read(base/'source.json')
 assert all(sha(ROOT/f)==h for f,h in sources['sources'].items())
for base,name in [(BASE,'profile_job.json'),(CORPUS,'capture_job.json')]:
 r=read(base/name);assert r['status']=='finished' and r['exit_code']==0
r=read(CORPUS/'verification_job.json');assert r['status']=='finished' and r['exit_code']==0
entry=(BASE/'status_entry.txt').read_text()
for f in ['PROJECT_STATUS.md','START_HERE.md','REMEDIATION_CHECKLIST.md','RESEARCH_ROADMAP.md','SUBMISSION_SPRINT.md','PAPER_WORKING_DRAFT.md','SIMPLE_NOTES.md','explanation.md']:assert entry in (ROOT/f).read_text(),f
plan=read(ROOT/'results/hardware/cpu_state_transition_v1/plan.json')
for row in plan['evidence'].values():assert sha(ROOT/row['path'])==row['sha256']
failed=read(ROOT/'results/hardware/pooled_controller_v1/verified_cost.json');assert not failed['pooled_controller_wholeloop_primary_passed']
# Historical measured-source and proof records remain unchanged after adding docs.
previous=ROOT/'results/hardware/pooled_controller_v1';old=read(previous/'verified_status.json')
for f,h in old['evidence_sha256'].items():assert sha(previous/f)==h,f
r=dict(verified=True,full_goal_complete=False,previous_turn='progress: measured remaining controller cost and added exact1600-transition corpus',all_owned_jobs_terminal=True,corpus_records=1600,cpu_transition_implemented=False,failed_candidate_stop_preserved=True,unit_tests_last_verified=459,source_and_evidence_hashes_match=True,production_defaults_unchanged=True,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'verified_diagnostic.json',CORPUS/'verified_corpus.json',ROOT/'results/hardware/cpu_state_transition_v1/plan.json',BASE/'status_entry.txt',BASE/'verify_status.py']})
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='evidence_sha256'},indent=2))
