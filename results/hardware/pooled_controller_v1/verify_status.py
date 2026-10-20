"""Read-only terminal evidence/source/status audit, no rerun of benchmarks."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=BASE/'verified_status.json';assert not out.exists()
for name in ['verified_correctness.json','verified_resource.json','verified_cost.json']:
 r=read(BASE/name);assert r['verified']
 for f,h in r['evidence_sha256'].items():assert sha(BASE/f)==h,(name,f)
for kind in ['gpu','resource','cost']:
 r=read(BASE/(kind+'_implementation.json'));assert sha(BASE/(kind+'_source.tar.gz'))==r['archive_sha256']
 with tarfile.open(BASE/(kind+'_source.tar.gz')) as t:
  sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
 assert sources==r['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
 job=read(BASE/(kind+'_job.json'));assert job['status']=='finished' and job['exit_code']==0
for name in ['attempt1_job.json','attempt2_job.json','tests_job.json','tests_v2_job.json']:assert read(BASE/name)['status']=='finished'
build=read(BASE/'native_build.json')
for prefix in ['pooled','controller']:assert sha(Path(build[prefix+'_binary']))==build[prefix+'_binary_sha256']
assert sha(ROOT/'experiments/native_controller.cpp')==build['controller_source_sha256']
for f,h in build['pooled_sources'].items():assert sha(ROOT/'experiments'/f)==h
fig=read(BASE/'figure_manifest.json')
assert sha(BASE/'plot_results.py')==fig['plot_source_sha256']
for f,h in fig['artifacts_sha256'].items():assert sha(BASE/f)==h
assert read(BASE/'verified_resource.json')['resource_primary_passed']
assert not read(BASE/'verified_cost.json')['pooled_controller_wholeloop_primary_passed']
assert len(read(BASE/'completion_ledger.json')['requirements'])==7
entry=(BASE/'status_entry.txt').read_text()
for f in ['PROJECT_STATUS.md','START_HERE.md','REMEDIATION_CHECKLIST.md','RESEARCH_ROADMAP.md','SUBMISSION_SPRINT.md','PAPER_WORKING_DRAFT.md']:
 assert entry in (ROOT/f).read_text(),f
 for word in ['459 tests pass','Stop pooled controller expansion/adoption','All owned jobs are terminal']:assert word in entry
r=dict(verified=True,full_goal_complete=False,ordinary_adoption=False,tests_passed=459,resource_passed=True,speed_passed=False,all_owned_jobs_terminal=True,all_archived_sources_and_binary_hashes_match=True,completion_requirements_preserved=7,disposition='Stop this candidate. Full P0/P1/P2 remains open; unsupported day estimate withdrawn.',evidence_sha256={p.name:sha(p) for p in BASE.iterdir() if p.is_file() and p!=out})
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='evidence_sha256'},indent=2))
