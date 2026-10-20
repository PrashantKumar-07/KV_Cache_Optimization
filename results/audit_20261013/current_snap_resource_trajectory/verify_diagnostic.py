"""Root-agent resource proof; not an independent researcher review."""
from pathlib import Path
import datetime,hashlib,json,sys,tarfile
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from provenance import model_artifact_identity
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
outpath=BASE/'verified_diagnostic.json';assert not outpath.exists()
p=json.loads((BASE/'plan.json').read_text());r=json.loads((BASE/'diagnostic.json').read_text())
assert r['passed'] and r['plan_sha256']==sha(BASE/'plan.json')
assert sha(BASE/'run_diagnostic.py')==p['runner_sha256']
assert sha(ROOT/p['source_archive'])==p['source_archive_sha256']
assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
with tarfile.open(ROOT/p['source_archive'],'r:gz') as tf:
 frozen={m.name:hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf if m.isfile()}
assert frozen==p['source_sha256']
assert model_artifact_identity(p['model'])==p['model_identity']
for name in ['prompt','teacher_stream']:assert sha(ROOT/p[name])==p[name+'_sha256']
parity=json.loads((ROOT/'results/audit_20261013/reusable_root_parity/parity.json').read_text())
assert r['input_ids_sha256']==parity['request_input_ids_sha256']['16384']
snap=r['arms']['snap'];native=r['arms']['native']
assert len(snap['boundaries'])==len(native['boundaries'])==51
assert native['predicted_token_ids']==parity['pairs']['16384']['candidate']['predicted_token_ids']
assert native['counts']==dict(cold_boundary_steps=50,cpu_append_calls=50,cpu_fallback_steps=0,cold_boundary_fallback_calls=1)
for step,b in enumerate(snap['boundaries']):
 assert len(b['layers'])==32 and all(l['visible_rows']==904+step and l['visible_heads']==32 for l in b['layers'])
 assert b['visible_tensor_payload_bytes']==32*2*32*(904+step)*128*2
 assert b['deduplicated_named_KV_storage_bytes']=={'cuda:0':b['visible_tensor_payload_bytes']}
for b in native['boundaries']:
 assert len(b['layers'])==32 and all(l['visible_heads']==8 and l['hot_rows']<=1024 and l['cold_total_rows']<=2048 for l in b['layers'])
 assert b['deduplicated_named_KV_storage_bytes']['cpu']==276824064
 assert b['gpu_reserved_bytes']>=b['gpu_allocated_bytes']
for item in [native['actual_pinned_pool'],r['retained_idle_pool']]:
 assert item['requested_storage_bytes']==item['allocator_block_bytes']==276824064
assert r['allocation_stats_at_exit']==r['final_allocation_stats']
stats=r['final_allocation_stats'];assert stats['live_allocations']==stats['live_requested_bytes']==0
events=stats['events'];assert len(events)==2 and [x['kind'] for x in events]==['allocate','release'] and events[0]['token']==events[1]['token']
summary={}
for name,arm in r['arms'].items():
 b=arm['boundaries'];summary[name]={
  'initial_named_KV_storage_bytes':b[0]['deduplicated_named_KV_storage_bytes'],
  'final_named_KV_storage_bytes':b[-1]['deduplicated_named_KV_storage_bytes'],
  'max_boundary_GPU_allocated_minus_model_bytes':max(x['gpu_allocated_bytes'] for x in b)-r['constant_model_storage_bytes'],
  'max_boundary_GPU_reserved_bytes':max(x['gpu_reserved_bytes'] for x in b),
  'initial_visible_rows_per_layer':[x['visible_rows'] for x in b[0]['layers']],
  'final_visible_rows_per_layer':[x['visible_rows'] for x in b[-1]['layers']],
  'max_CPU_canonical_metadata_bytes':max(x['cpu_canonical_metadata_bytes'] for x in b)}
out={'verified':True,'reviewer':'root-agent executable proof; independent review pending',
 'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':len(frozen),
 'summary':summary,'fifty_one_boundaries_verified_per_arm':True,'native_predictions_match_prior_fullmodel_parity':True,
 'one_actual_264MiB_root_active_and_idle_finally_zero':True,'resource_matched':False,'timing_valid':False,
 'scope':p['scope'],'limitations':['One prompt, teacherforced50steps, no quality or timing evidence',
 'Different selection algorithms, queryhead/KVhead representation, attendedwidth, CPU/GPU placement and growingSnapcapacity; no equalwholememory interpretation',
 'NamedKVstorage omits nonKVstate; CPUcanonical excludes Python/nativecontainer rounding and temporary scratch; GPUallocated excludes driver; boundary observations are not fullprefill/decode peaks',
 'Baseline adapter currently recomputes Q/K projections in capture hook; assess/remove redundant adapter work before headline latency comparison'],
 'evidence_sha256':{str(q.relative_to(ROOT)):sha(q) for q in [BASE/'plan.json',BASE/'run_diagnostic.py',BASE/'diagnostic.json',Path(__file__),ROOT/p['source_archive']]}}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
