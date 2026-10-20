"""CPU-only independent sequential root parity/lifetime verification."""
from pathlib import Path
import datetime,hashlib,json,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
from provenance import model_artifact_identity,runtime_identity

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
output=BASE/'verified_parity.json';assert not output.exists()
p=json.loads((BASE/'plan.json').read_text());r=json.loads((BASE/'parity.json').read_text())
assert r['passed'] is True and r['plan_sha256']==sha(BASE/'plan.json') and sha(BASE/'run_parity.py')==p['runner_sha256']
assert sha(ROOT/p['source_archive'])==p['source_archive_sha256'];frozen={}
with tarfile.open(ROOT/p['source_archive'],'r|gz') as tf:
 for m in tf:
  if m.isfile():
   h=hashlib.sha256();f=tf.extractfile(m)
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   frozen[m.name.removeprefix('./')]=h.hexdigest()
assert frozen==p['source_sha256'];assert all(sha(ROOT/name)==h for name,h in frozen.items())
assert runtime_identity()==r['runtime']
for key in ['prompt','teacher_stream']:assert sha(ROOT/p[key])==p[key+'_sha256']
assert r['model_identity']==p['model_identity']==model_artifact_identity(p['model'])
assert p['contexts']==[16384,8192] and set(r['pairs'])=={'16384','8192'}
fields={'hot_pos','cold_pos','cold_shadow','hot_k','hot_v','cold_k','cold_v','visible_k','visible_v',
 'hot_cumulative','hot_age','cold_last_access','cold_peek_streak','cold_was_hot'}
summary={};pointer=None;events=None
for n,context in enumerate(p['contexts']):
 pair=r['pairs'][str(context)];left,right=pair['control'],pair['candidate']
 assert left['trace']==right['trace'] and len(left['trace'])==51
 for frame in left['trace']:
  assert set(frame)=={str(i) for i in range(32)}
  for row in frame.values():
   assert set(row)==fields
   assert len(row['hot_pos'])==len(set(row['hot_pos']))<=1024
   assert len(row['cold_pos'])==len(set(row['cold_pos']))<=2048 and len(row['cold_shadow'])==len(row['cold_pos'])
   for field in fields-{'hot_pos','cold_pos','cold_shadow'}:assert len(row[field])==64;int(row[field],16)
 assert left['predicted_token_ids']==right['predicted_token_ids'] and len(left['predicted_token_ids'])==50
 for arm,row in pair.items():
  assert row['forwards']==51 and row['hooks_closed'] is True and row['lease_closed'] is True and row['owner_idle'] is True
  st=row['allocation_stats_after_request_reset']
  expected=0 if n==0 and arm=='control' else 1
  assert st['live_allocations']==expected and st['live_requested_bytes']==expected*276824064
  if expected:
   assert len(st['events'])==1 and st['events'][0]['kind']=='allocate' and st['events'][0]['bytes']==276824064
   if events is None:events=st['events']
   assert st['events']==events
 counts=right['counts'];assert counts['cold_boundary_steps']==counts['cpu_append_calls']==50 and counts['cpu_fallback_steps']==0
 assert counts['cold_boundary_fallback_calls']==1 and counts['bulk_payload_steps']+counts['bulk_payload_fallback_steps']==50
 if pointer is None:pointer=right['pool_pointer']
 assert pointer==right['pool_pointer']
 for key in ['pool_info','idle_pool_info']:
  info=right[key];assert info['requested_storage_bytes']==info['allocator_block_bytes']==276824064 and info['public_cuda_memory_type']==1 and info['allocated'] is True
 with np.load(BASE/f'{context}_control_logits.npz') as a,np.load(BASE/f'{context}_candidate_logits.npz') as b:
  assert set(a.files)==set(b.files)=={str(i) for i in range(51)}
  for i in range(51):
   x,y=a[str(i)],b[str(i)];assert x.shape==y.shape==(1,1,32000) and x.dtype==y.dtype==np.dtype('float32')
   assert np.isfinite(x).all() and np.isfinite(y).all() and x.tobytes()==y.tobytes()
   if i:assert int(x[0,-1].argmax())==left['predicted_token_ids'][i-1]
 summary[str(context)]={'finite_full_vocab_arrays_byte_exact':51,'paired32layer_trace_points':51,'five_metadata_hashes_exact':True,
  'predictions_recomputed_from_decode_logits':50,'counts':counts,'retained_idle_pinned_bytes':276824064}
f=r['lifecycle_failure_probe'];assert 'Captured prompt query shape' in f['error']
assert f['registered_layer_caches']==0 and f['attached_manager_caches']==1 and f['owner_busy_before_reset'] is True
assert f['attached_views_closed'] is True and f['owner_idle_after_reset'] is True and f['retained_root_pointer']==pointer
assert f['allocation_stats_after_reset']['live_allocations']==1 and f['allocation_stats_after_reset']['live_requested_bytes']==276824064
assert f['allocation_stats_after_reset']['events']==events
assert r['owner_leases_including_failure_probe']==3
final=r['allocation_stats_after_final_close'];assert final==r['allocation_stats_at_exit']
assert final['live_allocations']==final['live_requested_bytes']==0 and len(final['events'])==2
assert final['events'][0]==events[0] and final['events'][1]['kind']=='release'
assert final['events'][0]['token']==final['events'][1]['token'] and final['events'][0]['bytes']==final['events'][1]['bytes']==276824064
assert r['retained_idle_pool_info']['requested_storage_bytes']==r['retained_idle_pool_info']['allocator_block_bytes']==276824064
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(p['model']);ids=tok((ROOT/p['prompt']).read_text(),return_tensors='np')['input_ids']
for context in p['contexts']:assert hashlib.sha256(ids[:,:context].astype(np.int64,copy=False).tobytes()).hexdigest()==r['request_input_ids_sha256'][str(context)]
assert r['request_input_ids_sha256']['16384']!=r['request_input_ids_sha256']['8192']
out={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only_verifier':True,
 'source_file_count':len(frozen),'archive_plan_live_runtime_match':True,'model_artifacts_rehashed':len(r['model_identity']['artifact_sha256']),
 'summary':summary,'one_root_pointer_across_contexts_and_failed_attach':True,'request_input_ids_rehashed':True,
 'lifecycle_attached_failed_initialization_cleanup_verified':True,'single_allocation_release_pair':True,
 'retained_idle_requested_and_allocator_bytes':276824064,'zero_live_roots_after_final_close':True,
 'limits':['OneMistral/seed, two prefixcontexts ofsame consumedtext; not independentquality/generalization',
 'Creationthread/serialsameCUDAcontext only; no concurrentrequests or geometrymigration',
 'Persistent264MiB rootcounted whileidle; startup/provisioning is not servingTTFT and previousTTFTguard failure remains',
 'Diagnostic fullpayload copies/sync invalidate traffic/timing; no SOTA/physicalSTT/defaultadoption'],
 'evidence_sha256':{str(q.relative_to(ROOT)):sha(q) for q in [BASE/'plan.json',BASE/'run_parity.py',BASE/'parity.json',BASE/'source.tar.gz',Path(__file__)]}}
output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
