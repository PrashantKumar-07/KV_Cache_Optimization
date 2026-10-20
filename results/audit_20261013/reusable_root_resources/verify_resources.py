"""Independent steady-resource gate with explicit first-request/idle-root scope."""
from pathlib import Path
import datetime,hashlib,json,math,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
from provenance import model_artifact_identity,runtime_identity

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
outpath=BASE/'verified_resources.json';assert not outpath.exists(),'Preserve prior proof'
p=json.loads((BASE/'plan.json').read_text());r=json.loads((BASE/'timing.json').read_text())
prereq_path=ROOT/p['parity_prerequisite'];proof=json.loads(prereq_path.read_text())
assert proof['verified'] and proof['summary']['16384']['finite_full_vocab_arrays_byte_exact']==51
assert proof['single_allocation_release_pair'] and proof['zero_live_roots_after_final_close']
assert r['parity_prerequisite_sha256']==sha(prereq_path)
for name,h in proof['evidence_sha256'].items():assert sha(ROOT/name)==h,name
parity_base=ROOT/'results/audit_20261013/reusable_root_parity';parity=json.loads((parity_base/'parity.json').read_text())
parityplan=json.loads((parity_base/'plan.json').read_text());assert p['source_sha256']==parityplan['source_sha256']
assert r['passed'] is True and r['plan_sha256']==sha(BASE/'plan.json')
assert sha(BASE/'run_timing.py')==p['runner_sha256']
assert sha(ROOT/p['source_archive'])==p['source_archive_sha256'];frozen={}
with tarfile.open(ROOT/p['source_archive'],'r|gz') as tf:
 for m in tf:
  if m.isfile():
   h=hashlib.sha256();f=tf.extractfile(m)
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   frozen[m.name.removeprefix('./')]=h.hexdigest()
assert frozen==p['source_sha256'] and all(sha(ROOT/name)==h for name,h in frozen.items())
assert runtime_identity()==r['runtime']==parity['runtime']
for key in ['prompt','teacher_stream','quiet_helper']:assert sha(ROOT/p[key])==p[key+'_sha256']
checkpoint={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_count':len(frozen),'archive_plan_live_runtime_match':True,'runtime':r['runtime'],
 'plan_sha256':sha(BASE/'plan.json'),'runner_sha256':sha(BASE/'run_timing.py')}
cp=BASE/'source_attestation.json';assert not cp.exists();cp.write_text(json.dumps(checkpoint,indent=2)+'\n')
assert r['model_identity']==parity['model_identity']==model_artifact_identity(parityplan['model'])
settings=dict(p['settings']);deployment=settings.pop('deployment')
assert settings==dict(context=16384,decode=50,warmup=1,repeat=2,hot=1024,cold=2048,fraction=.75,quota=32,recent=128,dtype='BF16',last_logits=True,seed=0)
assert p['order']==['control','candidate','candidate','control']*3
quiet=r['quiet_observations'];assert len(quiet)==29
assert [v['stage'] for v in quiet]==['before_model_load']+['before_cold_request','after_cold_request']*2+['before_row','after_row']*12
uuid=quiet[0]['gpu'].split(',')[1].strip()
for q in quiet:
 assert q['quiet'] is True and q['unexpected_compute_processes']==[]
 assert q['gpu'].split(',')[0].strip()=='0' and q['gpu'].split(',')[1].strip()==uuid
 assert all(line.split(',')[0].strip()==uuid and int(line.split(',')[1])==r['pid'] for line in q['compute_processes'])
assert len(r['first_requests'])==2 and len(r['rows'])==12
reference=parity['pairs']['16384']['control']['predicted_token_ids']
static=r['constant_model_storage_bytes'];assert type(static) is int and static>0
pointer=None;events=None;lease_count=0;allocation_ms=None;memory=[];arms={'control':[],'candidate':[]};first=[]
sequence=r['first_requests']+r['rows']
for index,row in enumerate(sequence):
 isfirst=index<2;arm=['control','candidate'][index] if isfirst else p['order'][index-2]
 assert row['index']==index-2 and row['arm']==arm and row['cold_request'] is isfirst
 assert row['hooks_closed'] is True and row['lease_closed'] is True
 result=row['result'];assert result['timing_scope']=='end_to_end_model_policy_token'
 trials=result['trials'];assert len(trials)==(1 if isfirst else 2)
 for t in trials:
  assert t['decode_steps']==len(t['step_ms'])==len(t['forward_ms'])==len(t['predicted_token_ids'])==50
  assert t['predicted_token_ids']==reference
  for name in ['step_ms','forward_ms']:assert all(math.isfinite(v) and v>0 for v in t[name])
  for name in ['ttft_ms','prefill_peak_memory_gb','decode_peak_memory_gb']:assert math.isfinite(t[name]) and t[name]>0
  assert t['demoted']==t['paid_writes']+t['writes_saved']
  assert {k:t[k] for k in ['paid_writes','writes_saved','demoted','promoted']}==dict(paid_writes=2560,writes_saved=23872,demoted=26432,promoted=24832)
  if not isfirst:arms[arm].append(t)
 assert result['itl_ms']==round(float(np.median([v for t in trials for v in t['step_ms']])),3)
 assert result['ttft_ms']==round(float(np.median([t['ttft_ms'] for t in trials])),3)
 assert result['peak_memory_gb']==max(t['decode_peak_memory_gb'] for t in trials)
 assert result['prefill_peak_memory_gb']==max(t['prefill_peak_memory_gb'] for t in trials)
 if arm=='candidate':lease_count+=1 if isfirst else 3
 assert row['owner_leases']==lease_count
 st=row['exact_allocation_stats_after_cleanup'];expected=0 if index==0 else 1
 assert st['live_allocations']==expected and st['live_requested_bytes']==expected*276824064
 if expected:
  assert len(st['events'])==1 and st['events'][0]['kind']=='allocate' and st['events'][0]['bytes']==276824064
  if events is None:events=st['events']
  assert st['events']==events
  info=row['retained_idle_pool'];assert info['requested_storage_bytes']==info['allocator_block_bytes']==276824064
  assert info['allocated'] is True and info['public_cuda_memory_type']==1
  if pointer is None:pointer=row['retained_idle_pointer']
  assert row['retained_idle_pointer']==pointer
  if allocation_ms is None:allocation_ms=row['owner_allocation_ms']
  assert allocation_ms==row['owner_allocation_ms'] and math.isfinite(allocation_ms) and allocation_ms>0
 else:assert row['retained_idle_pool'] is None and row['retained_idle_pointer'] is None and row['owner_allocation_ms'] is None
 if arm=='candidate':
  c=row['counts'];assert c['cold_boundary_steps']==c['cpu_append_calls']==c['bulk_payload_steps']==50
  assert c['cpu_fallback_steps']==c['bulk_payload_fallback_steps']==0 and c['cold_boundary_fallback_calls']==1
  assert row['pool_pointer']==pointer
  pool=row['owned_pinned_pool'];assert pool['requested_storage_bytes']==pool['allocator_block_bytes']==276824064
  assert pool['allocated'] is True and pool['public_cuda_memory_type']==1
  gpu=round(max(t['decode_peak_memory_gb'] for t in trials)*1e9)-static;assert gpu>=0
  terms={'decode_GPU_excluding_static_model':gpu,'observed_main_pinned_allocator_block':pool['allocator_block_bytes'],
   'canonical_CPU_arrays':row['cpu_canonical_arrays_bytes'],'CPU_streak_array_peak':c['controller_cpu_streak_array_peak'],
   'CPU_address_array_peak':c['controller_address_array_peak'],
   'registered_two_times_cumulative_descriptors':2*(c['bulk_descriptor_h2d_bytes']+c['cold_boundary_descriptor_h2d_bytes'])}
  assert all(type(v) is int and v>=0 for v in terms.values())
  if not isfirst:memory.append({'row_index':row['index'],'terms_bytes':terms,'scoped_bytes':sum(terms.values())})
 if isfirst:first.append({'arm':arm,'ttft_ms':trials[0]['ttft_ms'],'median_itl_ms':float(np.median(trials[0]['step_ms'])),
  'decode_forwards':50,'owner_provisioning_ms_observation':row['owner_allocation_ms']})
assert lease_count==19 and r['owner_closed'] is True and r['final_owner_token_active'] is False
final=r['final_exact_allocation_stats'];assert final['live_allocations']==final['live_requested_bytes']==0 and len(final['events'])==2
assert final['events'][0]==events[0] and final['events'][1]['kind']=='release'
assert final['events'][0]['token']==final['events'][1]['token'] and final['events'][0]['bytes']==final['events'][1]['bytes']==276824064
summary={}
for arm,trials in arms.items():
 steps=[v for t in trials for v in t['step_ms']];assert len(steps)==600
 summary[arm]={'measured_decode_forwards':600,'median_itl_ms':float(np.median(steps)),
 'p95_itl_ms':float(np.quantile(steps,.95)),'median_ttft_ms':float(np.median([t['ttft_ms'] for t in trials])),
 'peak_decode_alloc_gb':max(t['decode_peak_memory_gb'] for t in trials),'peak_prefill_alloc_gb':max(t['prefill_peak_memory_gb'] for t in trials)}
ratios={name:summary['candidate'][metric]/summary['control'][metric] for name,metric in
 [('median_itl_ratio','median_itl_ms'),('p95_itl_ratio','p95_itl_ms'),('median_ttft_ratio','median_ttft_ms'),
 ('peak_decode_alloc_ratio','peak_decode_alloc_gb'),('peak_prefill_alloc_ratio','peak_prefill_alloc_gb')]}
scoped=max(v['scoped_bytes'] for v in memory)
guards={name:(scoped if name=='scoped_cache_gpu_cpu_bytes_max' else ratios[name.removesuffix('_max')])<=value for name,value in p['guards'].items()}
blocks=[]
for block in range(3):
 rows=r['rows'][4*block:4*block+4];med={}
 for arm in arms:med[arm]=float(np.median([v for row in rows if row['arm']==arm for t in row['result']['trials'] for v in t['step_ms']]))
 blocks.append({'block':block,'median_itl_ms':med,'candidate_control_ratio':med['candidate']/med['control']})
out={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_attestation':checkpoint,
 'model_artifacts_rehashed':len(r['model_identity']['artifact_sha256']),'quiet_observations_passed':29,
 'steady_summary':summary,'steady_ratios':ratios,'registered_steady_guards':guards,
 'steady_resource_advancement_passed':all(guards.values()),'decision':'advance internal steadyresource screen only' if all(guards.values()) else 'reject registered steadyresource advancement; preserve failure',
 'first_observed_requests_order_control_then_candidate':first,'first_request_observations_are_not_causal_cold_improvement':True,
 'first_requests_excluded_from_steady_guard_denominators':True,'candidate_owner_leases':19,'one_allocation_final_release_pair':True,
 'idle_persistent_pinned_bytes':276824064,'zero_live_roots_after_final_close':True,'all_predictions_match_verified_16K_parity':True,
 'closed_hooks_leases_and_native_path_counts_verified':True,'ABBA_blocks':blocks,'candidate_memory_rows':memory,
 'max_scoped_cache_bytes':scoped,'max_scoped_cache_MiB':scoped/2**20,'registered_memory_ceiling_bytes':p['guards']['scoped_cache_gpu_cpu_bytes_max'],
 'deployment':deployment,'scope':p['scope'],'memory_scope_formula':p['scope_formula'],'adopted':False,
 'limitations':['Ordered firstrequests have startup/kernelwarming/orderconfounding; not causal coldTTFT comparison or priorfreshallocatorguard pass',
 'Persistent264MiBroot is countedidle andactive; firstrequest includes oneoffprovisioning; no hidden capacity amortization',
 'One16K consumedtext/frozenteacherstream; no independentquality/SOTA/physicalSTT/defaultadoption',
 'ScopedGPU/CPUowner formula excludes hiddenhost/driver/RSS/container rounding;2xdescriptors is a registered conservative reservation, not measured simultaneous occupancy',
 'Quiet checks bracket rows, not continuousGPUutilization; creationthread/singlecontext serialrequests only'],
 'evidence_sha256':{str(q.relative_to(ROOT)):sha(q) for q in [BASE/'plan.json',BASE/'run_timing.py',BASE/'timing.json',ROOT/p['source_archive'],prereq_path,Path(__file__)]}}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['candidate_memory_rows','source_attestation','evidence_sha256']},indent=2))
