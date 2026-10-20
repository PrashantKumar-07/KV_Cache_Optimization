"""Independent read-only resource verifier; rejected guards remain rejected."""
from pathlib import Path
import datetime,hashlib,json,math,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
from provenance import model_artifact_identity,runtime_identity

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
output=BASE/'verified_resources.json';assert not output.exists(),'Preserve prior proof'
plan=json.loads((BASE/'plan.json').read_text());raw=json.loads((BASE/'timing.json').read_text())
prereq_path=ROOT/plan['parity_prerequisite']
prereq=json.loads(prereq_path.read_text());assert prereq['verified'] and prereq['finite_and_bytes_exact'] and prereq['exact_live_roots_zero_after_reset']
assert raw['parity_prerequisite_sha256']==sha(prereq_path)
for name,value in prereq['evidence_sha256'].items():assert sha(ROOT/name)==value,name
parity_base=ROOT/'results/audit_20261013/exact_controller'
parity=json.loads((parity_base/'parity.json').read_text())
parity_plan=json.loads((parity_base/'plan.json').read_text())
assert plan['source_sha256']==parity_plan['source_sha256']
assert raw['plan_sha256']==sha(BASE/'plan.json')
assert plan['runner_sha256']==sha(BASE/'run_timing.py')
archive_path=ROOT/plan['source_archive'];assert sha(archive_path)==plan['source_archive_sha256']
archive={}
with tarfile.open(archive_path,'r|gz') as tar:
 for member in tar:
  if member.isfile():
   h=hashlib.sha256();stream=tar.extractfile(member)
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
   archive[member.name.removeprefix('./')]=h.hexdigest()
assert archive==plan['source_sha256']
for name,value in archive.items():assert sha(ROOT/name)==value,name
for name in ['prompt','teacher_stream','quiet_helper']:assert sha(ROOT/plan[name])==plan[name+'_sha256']
live_runtime=runtime_identity();assert live_runtime==raw['runtime']==parity['runtime']
source_checkpoint={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_file_count':len(archive),'archive_equals_plan_equals_live':True,'runtime':live_runtime,
 'plan_sha256':sha(BASE/'plan.json'),'runner_sha256':sha(BASE/'run_timing.py')}
checkpoint=BASE/'independent_source_attestation.json';assert not checkpoint.exists()
checkpoint.write_text(json.dumps(source_checkpoint,indent=2)+'\n')
assert raw['model_identity']==parity['model_identity']==model_artifact_identity(raw['model_identity']['snapshot_path'])
assert raw['passed'] is True and len(raw['rows'])==12
expected_order=['control','candidate','candidate','control']*3
assert plan['order']==expected_order
assert plan['settings']==dict(context=16384,decode=50,warmup=1,repeat=2,hot=1024,cold=2048,fraction=.75,quota=32,recent=128,dtype='BF16',last_logits=True,seed=0)
q=raw['quiet_observations'];assert len(q)==25
assert [v['stage'] for v in q]==['before_model_load']+['before_row','after_row']*12
uuid=q[0]['gpu'].split(',')[1].strip()
for obs in q:
 assert obs['quiet'] is True and obs['unexpected_compute_processes']==[]
 assert obs['gpu'].split(',')[0].strip()=='0' and obs['gpu'].split(',')[1].strip()==uuid
 assert all(line.split(',')[0].strip()==uuid and int(line.split(',')[1])==raw['pid'] for line in obs['compute_processes'])
static=raw['constant_model_storage_bytes'];assert type(static) is int and static>0
arms={'control':[],'candidate':[]};memory=[];previous_events=[];allocation_checks=[]
reference=parity['arms']['control']['predicted_token_ids']
for i,row in enumerate(raw['rows']):
 arm=expected_order[i];assert row['index']==i and row['arm']==arm
 assert row['hooks_closed'] is True and row['lease_closed'] is True
 stats=row['exact_allocation_stats_after_cleanup']
 assert stats['live_allocations']==0 and stats['live_requested_bytes']==0
 events=stats['events'];assert events[:len(previous_events)]==previous_events
 newly=events[len(previous_events):];live={}
 for event in events:
  assert event['bytes']==276824064
  if event['kind']=='allocate':
   assert event['token'] not in live;live[event['token']]=event['bytes']
  else:
   assert event['kind']=='release' and live.pop(event['token'])==event['bytes']
 assert not live
 assert len(newly)==(6 if arm=='candidate' else 0)
 allocation_checks.append({'row_index':i,'new_exact_allocation_release_pairs':len(newly)//2,'live_roots_after_cleanup':0})
 previous_events=events
 result=row['result'];assert result['timing_scope']=='end_to_end_model_policy_token'
 trials=result['trials'];assert len(trials)==2
 for trial in trials:
  assert trial['decode_steps']==len(trial['step_ms'])==len(trial['forward_ms'])==len(trial['predicted_token_ids'])==50
  assert trial['predicted_token_ids']==reference
  for name in ['step_ms','forward_ms']:
   assert all(math.isfinite(v) and v>0 for v in trial[name])
  for name in ['ttft_ms','prefill_peak_memory_gb','decode_peak_memory_gb']:
   assert math.isfinite(trial[name]) and trial[name]>0
  assert trial['demoted']==trial['paid_writes']+trial['writes_saved']
  assert {k:trial[k] for k in ['paid_writes','writes_saved','demoted','promoted']}==dict(paid_writes=2560,writes_saved=23872,demoted=26432,promoted=24832)
  arms[arm].append(trial)
 assert result['itl_ms']==round(float(np.median([v for t in trials for v in t['step_ms']])),3)
 assert result['ttft_ms']==round(float(np.median([t['ttft_ms'] for t in trials])),3)
 assert result['peak_memory_gb']==max(t['decode_peak_memory_gb'] for t in trials)
 assert result['prefill_peak_memory_gb']==max(t['prefill_peak_memory_gb'] for t in trials)
 if arm=='candidate':
  c=row['counts']
  assert c['cold_boundary_steps']==c['cpu_append_calls']==c['bulk_payload_steps']==50
  assert c['cpu_fallback_steps']==c['bulk_payload_fallback_steps']==0
  assert c['cold_boundary_fallback_calls']==1 # initialization before CPU arena attachment
  pool=row['owned_pinned_pool']
  assert pool['allocated'] is True and pool['public_cuda_memory_type']==1
  assert pool['requested_storage_bytes']==276_824_064
  assert pool['allocator_block_bytes']==pool['requested_storage_bytes']
  gpu_bytes=round(max(t['decode_peak_memory_gb'] for t in trials)*1e9)-static
  assert gpu_bytes>=0
  terms={'decode_GPU_excluding_static_model':gpu_bytes,'observed_main_pinned_allocator_block':pool['allocator_block_bytes'],
   'canonical_CPU_arrays':row['cpu_canonical_arrays_bytes'],'CPU_streak_array_peak':c['controller_cpu_streak_array_peak'],
   'CPU_address_array_peak':c['controller_address_array_peak'],
   'registered_two_times_cumulative_descriptors':2*(c['bulk_descriptor_h2d_bytes']+c['cold_boundary_descriptor_h2d_bytes'])}
  assert all(type(v) is int and v>=0 for v in terms.values())
  memory.append({'row_index':i,'terms_bytes':terms,'scoped_bytes':sum(terms.values()),'requested_pool_bytes':pool['requested_storage_bytes']})
summary={}
for arm,trials in arms.items():
 steps=[v for trial in trials for v in trial['step_ms']];assert len(steps)==600
 summary[arm]={'measured_decode_forwards':len(steps),'median_itl_ms':float(np.median(steps)),
  'p95_itl_ms':float(np.quantile(steps,.95)),'median_ttft_ms':float(np.median([t['ttft_ms'] for t in trials])),
  'peak_decode_alloc_gb':max(t['decode_peak_memory_gb'] for t in trials),
  'peak_prefill_alloc_gb':max(t['prefill_peak_memory_gb'] for t in trials)}
ratios={name:summary['candidate'][metric]/summary['control'][metric] for name,metric in
 [('median_itl_ratio','median_itl_ms'),('p95_itl_ratio','p95_itl_ms'),('median_ttft_ratio','median_ttft_ms'),
  ('peak_decode_alloc_ratio','peak_decode_alloc_gb'),('peak_prefill_alloc_ratio','peak_prefill_alloc_gb')]}
scoped=max(v['scoped_bytes'] for v in memory)
guards={name:(scoped if name=='scoped_cache_gpu_cpu_bytes_max' else ratios[name.removesuffix('_max')])<=value
 for name,value in plan['guards'].items()}
blocks=[]
for block in range(3):
 rows=raw['rows'][block*4:block*4+4];byarm={}
 for arm in arms:
  steps=[x for r in rows if r['arm']==arm for t in r['result']['trials'] for x in t['step_ms']]
  byarm[arm]=float(np.median(steps))
 blocks.append({'block':block,'median_itl_ms':byarm,'candidate_control_ratio':byarm['candidate']/byarm['control']})
out={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_attestation':source_checkpoint,'model_artifacts_rehashed':len(raw['model_identity']['artifact_sha256']),
 'parity_prerequisite_sha256':sha(prereq_path),'quiet_observations_passed':25,'all_predictions_match_parity_prerequisite':True,
 'hooks_leases_closed':True,'candidate_native_cold_CPU_append_steps_per_final_trial':50,
 'candidate_CPU_bulk_fallback_steps':0,'expected_prefill_cold_boundary_fallback_calls':1,
 'summary':summary,'ratios':ratios,'registered_guards':guards,'ABBA_blocks':blocks,
 'candidate_memory_rows':memory,'exact_allocator_row_checks':allocation_checks,'total_exact_allocation_release_pairs':len(previous_events)//2,'max_scoped_cache_bytes':scoped,'max_scoped_cache_MiB':scoped/2**20,
 'registered_memory_ceiling_bytes':plan['guards']['scoped_cache_gpu_cpu_bytes_max'],
 'resource_advancement_passed':all(guards.values()),
 'decision':'advance internal resource screen only' if all(guards.values()) else 'reject registered resource advancement; preserve successful timing and failed memory guard separately',
 'adopted':False,'scope':plan['scope'],'memory_scope_formula':plan['scope_formula'],
 'limitations':['One reused16K teacherforced input; no quality/generalization/SOTA claim',
  'PyTorch GPUallocation includes model/workspaces; scoped GPUterm subtracts deduplicated staticmodelstorage',
  'Observed CUDAhost allocator block, not requested pool storage, is included; Python/native container rounding, hidden cached host/driver blocks andRSS excluded',
  '2x cumulative descriptor traffic is a preregistered conservative reservation, not observed simultaneous occupancy',
  'Quiet checks bracket rows, not continuous GPU utilization; CPU DRAM is not physicalSTT',
  'Native counts are finaltrial counts; warmup is excluded from600 measured forwards/arm'],
 'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [BASE/'plan.json',BASE/'run_timing.py',BASE/'timing.json',archive_path,prereq_path,Path(__file__)]}}
output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['candidate_memory_rows','evidence_sha256','source_attestation']},indent=2))
