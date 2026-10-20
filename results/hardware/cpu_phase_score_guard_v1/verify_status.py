"""Terminal current-source audit plus explicit historical scorer delta."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
out=BASE/'verified_status.json';assert not out.exists()
for name in ['verified_correctness.json','verified_resource.json','verified_cost.json','verified_geometry.json']:
 r=read(BASE/name);assert r['verified']
 for n,h in r['evidence_sha256'].items():assert sha(BASE/n)==h,(name,n)
for kind in ['gpu','resource','cost']:
 f=read(BASE/(kind+'_implementation.json'));assert sha(BASE/(kind+'_source.tar.gz'))==f['archive_sha256']
 with tarfile.open(BASE/(kind+'_source.tar.gz')) as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
 assert sources==f['sources'] and all(sha(ROOT/n)==h for n,h in sources.items())
 j=read(BASE/(kind+'_job.json'));assert j['status']=='finished' and j['exit_code']==0
fallback=ROOT/'results/hardware/cpu_phase_append_fallback_v1';fp=read(fallback/'verified_fallback.json');assert fp['verified'] and fp['tests_passed']==498
for n,h in fp['evidence_sha256'].items():assert sha(fallback/n)==h,n
historical={}
for dirname,proofname in [('cpu_phase_boundaries_v1','verified_boundaries.json'),('cpu_phase_geometry_v1','verified_geometry.json')]:
 old=ROOT/'results/hardware'/dirname;r=read(old/proofname);assert r['verified']
 for n,h in r['evidence_sha256'].items():assert sha(old/n)==h,(dirname,n)
 f=read(old/'gpu_implementation.json');assert sha(old/'gpu_source.tar.gz')==f['archive_sha256']
 with tarfile.open(old/'gpu_source.tar.gz') as t:
  archived={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
 assert {n:hashlib.sha256(v).hexdigest() for n,v in archived.items()}==f['sources']
 delta=[n for n,h in f['sources'].items() if sha(ROOT/n)!=h];assert delta==['experiments/batched_score_policy.py']
 original=archived[delta[0]].decode();needle='        if keep != list(range(len(cache.sram_cum))) or not keep:'
 guard='        if weight.ndim != 4 or weight.shape[0] != 1 or weight.shape[2] != 1:\n            return TieredKVPolicy.update_scores(policy, past, weights)\n'
 assert original.count(needle)==1 and original.replace(needle,guard+needle)==(ROOT/delta[0]).read_text()
 historical[dirname]=dict(proof_sha256=sha(old/proofname),source_delta=delta,delta_is_only_single_query_guard=True)
c=read(BASE/'cpu_build.json');assert sha(Path(c['binary']))==c['binary_sha256'] and sha(ROOT/'experiments/native_cpu_state.cpp')==c['source_sha256']
p=read(BASE/'pooled_build.json')
for prefix in ['pooled','controller']:assert sha(Path(p[prefix+'_binary']))==p[prefix+'_binary_sha256']
assert sha(ROOT/'experiments/native_controller.cpp')==p['controller_source_sha256']
for n,h in p['pooled_sources'].items():assert sha(ROOT/'experiments'/n)==h
fig=read(BASE/'figures/manifest.json');assert fig['component_proof_sha256']==sha(BASE/'verified_cost.json') and fig['geometry_proof_sha256']==sha(BASE/'verified_geometry.json') and fig['plot_source_sha256']==sha(BASE/'plot_results.py')
for n,h in fig['artifacts_sha256'].items():assert sha(BASE/'figures'/n)==h,n
assert read(BASE/'tests_job.json')['tests_passed']==498 and '498 passed' in (BASE/'suite.log').read_text()
assert read(BASE/'verified_resource.json')['resource_primary_passed'] and read(BASE/'verified_cost.json')['cpu_phase_controller_wholeloop_primary_passed']
assert not read(BASE/'verified_geometry.json')['equal_resource_P0_passed']
ledger=read(BASE/'completion_ledger.json');assert len(ledger['requirements'])==7 and not ledger['full_goal_complete']
entry=(BASE/'status_entry.txt').read_text()
for name in ['PROJECT_STATUS.md','START_HERE.md','SUBMISSION_SPRINT.md','explanation.md','EXPERIMENT_LOG.md','SIMPLE_NOTES.md','PAPER_WORKING_DRAFT.md','REMEDIATION_CHECKLIST.md','RESEARCH_ROADMAP.md']:assert entry in (ROOT/name).read_text(),name
assert 'All owned jobs are terminal' in entry and '498 tests pass' in entry and '58.76 to54.12' in entry
result=dict(verified=True,tests_passed=498,current_correctness_passed=True,current_resource_passed=True,current_ordinary_control_speed_passed=True,current_real_multi_token_fallback_passed=True,historical_boundary_proofs_preserved=True,historical_source_delta=historical,current_equal_resource_comparison_passed=False,all_owned_jobs_terminal=True,native_binaries_verified=True,figures_verified=True,full_requirements_preserved=7,ordinary_adoption=False,full_goal_complete=False,next='Legitimate equal-resource/comparable-width comparison design remains required; fixed capacities cannot match growing faithful baseline. Independent quality/physical hardware/generality/final joint claim remain open.',evidence_sha256={p.name:sha(p) for p in BASE.iterdir() if p.is_file() and p!=out})
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','historical_source_delta']},indent=2))
