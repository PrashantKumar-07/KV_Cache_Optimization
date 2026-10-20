"""Read-only paired replay and policy-profile attribution proof."""
import hashlib,json,pstats,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=BASE/'verified_diagnostic.json';assert not out.exists()
plan=read(BASE/'plan.json');source=read(BASE/'source.json')
assert source['plan_sha256']==sha(BASE/'plan.json') and source['archive_sha256']==sha(BASE/'source.tar.gz')
with tarfile.open(BASE/'source.tar.gz') as t:
 archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert archived==source['sources'] and all(sha(ROOT/f)==h for f,h in archived.items())
assert read(BASE/'profile_job.json')['exit_code']==0
failed=ROOT/'results/hardware/pooled_controller_v1/verified_cost.json'
assert sha(failed)==plan['failed_cost_sha256'] and not read(failed)['pooled_controller_wholeloop_primary_passed']
metrics={}
for arm in ['control','candidate']:
 assert sha(ROOT/plan[arm])==plan[arm+'_sha256']
 raw=read(BASE/(arm+'.json'));ref=read(ROOT/plan[arm])
 assert raw['passed'] and raw['frozen50predictions_state_counters_payload_calls_replayed'] and raw['hooks_restored'] and raw['owned_host_after_cleanup_bytes']==0
 assert raw['state_sha256']==ref['state_sha256'] and raw['counters']==ref['counters']
 assert raw['profiled_timing_not_latency_gate']['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
 for f,v in raw['payload'].items():assert v==ref['prototype_counts'][f]
 for f in ['native_h2d_calls','native_d2h_calls']:assert raw[f]==ref['prototype_counts'][f]
 assert len(raw['geometry'])==50 and all(v['saturated_layers']==0 for v in raw['geometry'])
 assert len(raw['quiet_observations'])==2 and all(v['quiet'] for v in raw['quiet_observations'])
 if arm=='candidate':assert raw['lease_closed'] and raw['pooled_counts']==dict(prepare=50,compact=200,descriptors=6457600,arena=417792,backups=1600,plans=50,fallbacks=0)
 stats=pstats.Stats(str(BASE/(arm+'.prof')))
 rows=[]
 for (file,line,name),(primitive,calls,exclusive,inclusive,callers) in stats.stats.items():
  if ((name in ['_prepare_decode_reads','_visible_tuple','_compact_or_queue','native_prepare','native_backups','controller_live','reserve_visible','_extend_hot_pair','set','get','compact','prepare','kernel','evict_and_demote'] and any(part in file for part in ['pooled','native_controller','ownership','raw_input','reserved','reuse','batched','production','fused_eviction'])) or 'tieredkv_controller_v1.backups' in name):
   rows.append(dict(file=file,line=line,name=name,calls=calls,exclusive_ms_per_step=exclusive*1000/50,inclusive_ms_per_step=inclusive*1000/50))
 metrics[arm]=dict(total_calls=stats.total_calls,total_profile_seconds=stats.total_tt,rows=rows)
backup=[r for r in metrics['candidate']['rows'] if r['name']=='native_backups'];assert len(backup)==1 and backup[0]['calls']==1600
initial=[r for r in metrics['candidate']['rows'] if r['name']=='kernel' and r['file'].endswith('/pooled_state.py')];assert len(initial)==1 and initial[0]['calls']==1
result=dict(verified=True,exact_frozen50_both_arms=True,metrics=metrics,candidate_backup_instrumented_ms_per_step=backup[0]['inclusive_ms_per_step'],candidate_first_extension_load_seconds=initial[0]['inclusive_ms_per_step']*50/1000,saturation_layers_observed=0,standalone_backup_batching_justified=False,decision='Do not retry the failed candidate or propose isolated backup batching as a >=4ms wholeloop fix. Next redesign must reduce broad CPU state-transition overhead and validate full phase ordering with fixed replay inputs.',limits=plan['limits']+' First extension load is a cold-start cost already excluded by warmup in the acceptance test. Do not add overlapping inclusive ranges or interpret these profiles as speed gates. No saturated-cold behavior covered by these50 steps.',full_goal_complete=False,evidence_sha256={f:sha(BASE/f) for f in ['plan.json','source.json','source.tar.gz','profile_job.json','profile.log','profile_pair.py','control.json','candidate.json','control.prof','candidate.prof','control.prof.txt','candidate.prof.txt','verify_diagnostic.py']})
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','evidence_sha256']},indent=2))
