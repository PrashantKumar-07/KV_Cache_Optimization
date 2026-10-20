"""Read-only acceptance and artifact checks; never converts stage gain to model gain."""
import hashlib,json,re
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
resource=read(B/'verified_resources.json');assert resource['passed'] and resource['tests_passed']==518
for f,h in resource['evidence_sha256'].items():assert sha(R/f)==h,f
ledger=read(B/'completion_ledger.json');assert not ledger['full_goal_complete'] and len(ledger['requirements'])==7
for d in ledger['current_evidence'].values():assert sha(R/d['path'])==d['sha256'],d['path']
for name in ['tests','gpu','parity_4096_12','parity_16384_50','failure','resource']:
 j=read(B/(name+'_job.json'));assert j['status']=='finished' and j['exit_code']==0
aborted=read(B/'cost_attempt1/cost.json');assert not aborted['passed'] and aborted['rows']==[] and not aborted['quiet_observations'][0]['quiet']
assert read(B/'cost_waiting.json')['measured_rows']==0 and not (B/'verified_cost.json').exists()
for f in ['PROJECT_STATUS.md','START_HERE.md','README.md','PAPER_CURRENT_EVIDENCE.md']:
 for target in re.findall(r'\]\(([^)]+)\)',(R/f).read_text()):
  if not target.startswith(('http:','https:','#','mailto:')):assert (R/target.split('#')[0]).exists(),(f,target)
files=[R/f for f in ['experiments/cold_boundary.py','experiments/cold_boundary_policy.py','experiments/native_cold_boundary.cpp','experiments/native_cold_boundary.cu','tests/test_cold_boundary.py','tests/test_cold_boundary_policy.py','PROJECT_STATUS.md','START_HERE.md','README.md','PAPER_CURRENT_EVIDENCE.md']]+[B/'completion_ledger.json',B/'verified_resources.json',B/'cost_plan.json',B/'verify_checkpoint.py']
output=B/'verified_checkpoint.json';assert not output.exists();output.write_text(json.dumps(dict(verified=True,tests_passed=518,exact_integrated_replay_and_cleanup=True,scope_mib=resource['scope_mib'],macro_state='waiting_for_quiet_gpu_zero_measured_rows',all_owned_jobs_terminal=True,all_original_requirement_groups_preserved=True,full_goal_complete=False,current_files_sha256={str(p.relative_to(R)):sha(p) for p in files},limits='Resource scope excludes unowned host containers/driver allocations. Stage timings are not model gain. GPU quiet guard aborted before model load; previous measured failures remain closed. No adoption/quality/physical hardware claim.'),indent=2)+'\n');print(output)
