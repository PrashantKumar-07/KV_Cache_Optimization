"""Final read-only status/source/evidence audit; no benchmark rerun."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=BASE/'verified_status.json';assert not out.exists()
proof=read(BASE/'verified_stage.json');assert proof['verified'] and proof['tests_passed']==489 and not proof['full_goal_complete']
for f,h in proof['evidence_sha256'].items():assert sha(BASE/f)==h,f
for name in ['stage_implementation.json','gpu_implementation.json']:
 source=read(BASE/name);assert all(sha(ROOT/f)==h for f,h in source['sources'].items())
for name in ['attempt1_job.json','focused_v2_job.json','suite_job.json','suite_v2_job.json','suite_v3_job.json','stage_v1_replay_job.json','replay_job.json','gpu_job.json']:assert read(BASE/name)['status']=='finished'
entry=(BASE/'status_entry.txt').read_text()
for f in ['PROJECT_STATUS.md','START_HERE.md','REMEDIATION_CHECKLIST.md','RESEARCH_ROADMAP.md','SUBMISSION_SPRINT.md','PAPER_WORKING_DRAFT.md','SIMPLE_NOTES.md','explanation.md']:assert entry in (ROOT/f).read_text(),f
r=dict(verified=True,full_goal_complete=False,all_owned_jobs_terminal=True,tests_passed=489,focused_phase_tests=30,recorded_layer_transitions=1600,gpu_handoff_cases=5,controller_integrated=False,production_defaults_unchanged=True,ready_for_controller_integration=True,limits='Native phase and synthetic GPU handoff only; full-model native KV, memory, speed, quality and physical hardware gates remain open.',evidence_sha256={p.name:sha(p) for p in BASE.iterdir() if p.is_file() and p!=out})
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='evidence_sha256'},indent=2))
