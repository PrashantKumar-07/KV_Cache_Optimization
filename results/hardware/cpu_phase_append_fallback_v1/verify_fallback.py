"""Independent real GPU multi-token fallback proof with preserved failed attempts."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
out=BASE/'verified_fallback.json';assert not out.exists()
f=read(BASE/'gpu_implementation.json');p=read(BASE/'plan.json');r=read(BASE/'gpu.json')
assert sha(BASE/'plan.json')==f['plan_sha256']==r['plan_sha256'] and sha(BASE/'gpu_source.tar.gz')==f['archive_sha256']
with tarfile.open(BASE/'gpu_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert sources==f['sources'] and all(sha(ROOT/n)==h for n,h in sources.items())
assert read(BASE/'gpu_job.json')['exit_code']==0 and read(BASE/'tests_job.json')['tests_passed']==498 and '498 passed' in (BASE/'suite.log').read_text()
assert read(BASE/'attempt1_job.json')['exit_code']==read(BASE/'attempt2_job.json')['exit_code']==1
for version in [1,2]:
 old=read(BASE/('attempt'+str(version)+'_gpu_implementation.json'))
 assert old['plan_sha256']==sha(BASE/('attempt'+str(version)+'_plan.json')) and old['archive_sha256']==sha(BASE/('attempt'+str(version)+'_gpu_source.tar.gz'))
assert r['passed'] and r['hooks_leases_and_cache_owners_released']
a,b=r['control'],r['candidate']
for n in ['frames','prototype_counts','native_h2d_calls','native_d2h_calls']:assert a[n]==b[n],n
assert len(a['frames'])==7 and all(len(x['layers'])==4 for x in a['frames'])
assert b['append_fast_updates']==16 and b['append_fallback_updates']==8
assert b['dispatch']==[dict(native=1,fallback=0,attached=True),dict(native=1,fallback=1,attached=True),dict(native=2,fallback=1,attached=True),dict(native=2,fallback=2,attached=True),dict(native=3,fallback=2,attached=True),dict(native=4,fallback=2,attached=True)]
r=dict(verified=True,tests_passed=498,focused_tests=15,real_Mistral_GPU=True,append_sizes=p['settings']['append_sizes'],native_phase_steps=4,multi_token_fallback_steps=2,exact_logit_KV_metadata_and_slot_boundaries=7,owned_hooks_leases_released=True,failed_attempts_preserved=2,full_goal_complete=False,ordinary_adoption=False,limits=p['limits'],evidence_sha256={n:sha(BASE/n) for n in ['plan.json','gpu_implementation.json','gpu_source.tar.gz','gpu_job.json','gpu.json','gpu.log','suite.log','focused.log','tests_job.json','regression_before.log','attempt1_job.json','attempt2_job.json','check_gpu.py','verify_fallback.py']})
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='evidence_sha256'},indent=2))
