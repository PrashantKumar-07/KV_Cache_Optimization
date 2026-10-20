"""Frozen warmed diagnostic proof; nested times are not additive savings."""
import hashlib,json,pstats,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=read(B/'profile_plan.json');m=read(B/'profile_implementation.json');d=read(B/'profile.json');j=read(B/'profile_job.json')
assert j['status']=='finished' and j['exit_code']==0 and d['passed'] and d['owned_hooks_leases_pool_released']
assert d['calls']==dict(policy=50,scores=50) and d['bulk_steps']==50 and d['bulk_fallback_steps']==0 and not d['valid_timing_gate']
assert sha(B/'profile_plan.json')==m['plan_sha256']==d['plan_sha256'] and sha(B/'profile_source.tar.gz')==m['archive_sha256']
with tarfile.open(B/'profile_source.tar.gz') as t:
 for f,h in m['sources'].items():assert sha(R/f)==hashlib.sha256(t.extractfile(f).read()).hexdigest()==h
ref=read(R/p['reference']);assert sha(R/p['reference'])==p['reference_sha256'] and d['state_sha256']==ref['state_sha256'] and d['counters']==ref['counters']
for k,v in d['native'].items():assert v==ref['prototype_counts'][k]
rows=[]
for scope in ['policy','scores']:
 stats=pstats.Stats(str(B/(scope+'.prof')))
 for (file,line,name),(primitive,calls,selftime,total,callers) in stats.stats.items():
  if name in ['_prepare_decode_reads','_visible_tuple','controller_live','native_prepare','execute','get','batched_update','address_tensor']:
   rows.append(dict(scope=scope,file=file,line=line,name=name,calls=calls,self_ms_per_step=1000*selftime/50,cumulative_ms_per_step=1000*total/50))
rows.sort(key=lambda x:x['cumulative_ms_per_step'],reverse=True)
out=B/'verified_profile.json';assert not out.exists();out.write_text(json.dumps(dict(verified=True,warmed_16K50_exact=True,no_resource_observer_in_profile=True,calls=d['calls'],attribution=rows,limits=p['limits']+' Nested cumulative times overlap; profiler perturbs dispatch and GPU synchronization. Do not add these costs or treat them as savings.',evidence_sha256={f:sha(B/f) for f in ['profile_plan.json','profile_implementation.json','profile_source.tar.gz','profile_job.json','profile.json','policy.prof','scores.prof']},full_goal_complete=False),indent=2)+'\n');print('Verified warmed profile:',len(rows),'attribution rows')
