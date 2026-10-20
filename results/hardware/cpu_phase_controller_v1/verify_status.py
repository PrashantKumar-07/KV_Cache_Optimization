"""Terminal component proof audit, preserving full completion requirements."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=BASE/'verified_status.json';assert not out.exists()
for kind in ['correctness','resource','cost','wider','recycling','generation']:
 r=read(BASE/('verified_'+kind+'.json'));assert r['verified']
 for n,h in r['evidence_sha256'].items():assert sha(BASE/n)==h,(kind,n)
for kind in ['gpu','resource','cost','wider','recycling','generation']:
 f=read(BASE/(kind+'_implementation.json'));assert f['archive_sha256']==sha(BASE/(kind+'_source.tar.gz'))
 with tarfile.open(BASE/(kind+'_source.tar.gz')) as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
 assert sources==f['sources'] and all(sha(ROOT/n)==h for n,h in sources.items())
 j=read(BASE/(kind+'_job.json'));assert j['status']=='finished' and j['exit_code']==0
c=read(BASE/'cpu_build.json');assert sha(Path(c['binary']))==c['binary_sha256'] and sha(ROOT/'experiments/native_cpu_state.cpp')==c['source_sha256']
p=read(BASE/'pooled_build.json')
for prefix in ['pooled','controller']:assert sha(Path(p[prefix+'_binary']))==p[prefix+'_binary_sha256']
assert sha(ROOT/'experiments/native_controller.cpp')==p['controller_source_sha256']
for n,h in p['pooled_sources'].items():assert sha(ROOT/'experiments'/n)==h
assert read(BASE/'tests_job.json')['tests_passed']==496 and '496 passed' in (BASE/'suite.log').read_text()
assert read(BASE/'verified_resource.json')['resource_primary_passed'] and read(BASE/'verified_cost.json')['cpu_phase_controller_wholeloop_primary_passed']
ledger=read(BASE/'completion_ledger.json');assert len(ledger['requirements'])==7 and not ledger['full_goal_complete']
for n in ['PROJECT_STATUS.md','START_HERE.md','SUBMISSION_SPRINT.md','explanation.md','EXPERIMENT_LOG.md','SIMPLE_NOTES.md','PAPER_WORKING_DRAFT.md','REMEDIATION_CHECKLIST.md','RESEARCH_ROADMAP.md']:
 s=(ROOT/n).read_text();assert '496 tests pass' in s and '58.93 to54.09' in s and 'All owned jobs are terminal' in s
result=dict(verified=True,tests_passed=496,correctness_passed=True,resource_passed=True,ordinary_control_speed_passed=True,wider_contexts_passed=True,cold_recycling_passed=True,development_free_generation_passed=True,all_owned_jobs_terminal=True,archived_sources_and_native_binaries_match=True,completion_requirements_preserved=7,ordinary_adoption=False,full_goal_complete=False,next='Current faithful baseline at actual matched resources/attention geometry, independent quality, and broader native fallback/exclusive boundaries; physical STT remains external.',evidence_sha256={n.name:sha(n) for n in BASE.iterdir() if n.is_file() and n!=out})
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
