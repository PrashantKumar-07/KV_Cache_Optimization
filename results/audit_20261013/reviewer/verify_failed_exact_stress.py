"""CPU diagnostic verification of rejected stress cases; never emits parity pass."""
from pathlib import Path
import datetime,hashlib,json,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from provenance import model_artifact_identity,runtime_identity

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
names=['exact_stress_8192_cold256','exact_stress_31500_cold2048'];records={}
for name in names:
 base=ROOT/'results/audit_20261013'/name
 plan=json.loads((base/'plan.json').read_text());raw=json.loads((base/'parity.json').read_text())
 assert raw['passed'] is False and raw['plan_sha256']==sha(base/'plan.json')
 assert plan['runner_sha256']==sha(base/'run_parity.py')
 for key in ['prompt','teacher_stream']:assert sha(ROOT/plan[key])==plan[key+'_sha256']
 assert sha(ROOT/plan['source_archive'])==plan['source_archive_sha256']
 records[name]=(base,plan,raw)
p0=records[names[0]][1];p1=records[names[1]][1]
assert p0['source_sha256']==p1['source_sha256']
arch={}
with tarfile.open(ROOT/p0['source_archive'],'r|gz') as tf:
 for m in tf:
  if m.isfile():
   h=hashlib.sha256();f=tf.extractfile(m)
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   arch[m.name.removeprefix('./')]=h.hexdigest()
assert arch==p0['source_sha256']
for p,h in arch.items():assert sha(ROOT/p)==h,p
runtime=runtime_identity();assert all(raw['runtime']==runtime for _,_,raw in records.values())
checkpoint={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_count':len(arch),'archive_plan_live_equal':True,'runtime':runtime,
 'cases':{name:{'plan_sha256':sha(base/'plan.json'),'runner_sha256':sha(base/'run_parity.py'),
 'raw_sha256':sha(base/'parity.json'),'prompt_sha256':plan['prompt_sha256'],'settings':plan['settings']} for name,(base,plan,raw) in records.items()}}
cp=ROOT/'results/audit_20261013/reviewer/failed_stress_source_attestation.json';assert not cp.exists()
cp.write_text(json.dumps(checkpoint,indent=2)+'\n')
print('SOURCE_ATTESTATION_COMPLETE',flush=True)
identity=records[names[0]][2]['model_identity']
assert all(raw['model_identity']==identity for _,_,raw in records.values())
assert model_artifact_identity(identity['snapshot_path'])==identity
reports={}
base,plan,raw=records[names[0]];left,right=raw['arms']['control'],raw['arms']['candidate']
assert left['trace']==right['trace'] and len(left['trace'])==51
live=[]
for step,frame in enumerate(right['trace']):
 assert set(frame)=={str(i) for i in range(32)}
 counts=[sum(not x for x in row['cold_shadow']) for row in frame.values()]
 assert all(len(row['hot_pos'])<=plan['settings']['hot'] and len(row['cold_pos'])<=plan['settings']['cold'] for row in frame.values())
 live.append({'tracepoint':step,'minimum_live_across_layers':min(counts),'maximum_live_across_layers':max(counts)})
assert left['predicted_token_ids']==right['predicted_token_ids'] and len(left['predicted_token_ids'])==50
for row in [left,right]:
 assert row['forwards']==51 and row['hooks_closed'] is True
 st=row['allocation_stats_after_reset'];assert st['live_allocations']==st['live_requested_bytes']==0
assert right['cpu_append_calls']==50 and right['cpu_fallback_steps']==0 and right['cold_steps']==41
with np.load(base/'control_logits.npz') as a,np.load(base/'candidate_logits.npz') as b:
 assert set(a.files)==set(b.files)=={str(i) for i in range(51)}
 for i in range(51):
  x,y=a[str(i)],b[str(i)]
  assert x.shape==y.shape==(1,1,32000) and x.dtype==y.dtype==np.dtype('float32')
  assert np.isfinite(x).all() and np.isfinite(y).all() and x.tobytes()==y.tobytes()
  if i:assert int(x[0,-1].argmax())==left['predicted_token_ids'][i-1]
assert raw['trace_matches'] and raw['prediction_matches'] and raw['full_logits_bitwise_match'] and raw['max_logit_abs_difference']==0
reports[names[0]]={'diagnostic_verified':True,'registered_parity_gate_passed':False,
 'reason':'native visibility coverage41 instead of required50; never loosen gate',
 'finite_full_vocab_bytes_exact_arrays':51,'payload_position_shadow_trace_points_each32layers':51,
 'decoded_predictions_recomputed_from_logits':50,'native_CPU_append_steps':50,'CPU_fallback_steps':0,
 'native_cold_visibility_steps':41,'inferred_non_native_visibility_steps':9,
 'successful_hooks_closed':True,'exact_live_roots_zero_after_reset':True,
 'post_state_live_counts':live,'empty_live_explanation_supported':False,
 'fallback_reason':'Unrecorded: all post-state traces have positive live count in every layer. Could be an unsupported geometry/ownership boundary; which rejection condition fired cannot be proved from hashes/positions without strides/owner/version/branch records.',
 'allocation_events':right['allocation_stats_after_reset']['events']}
base,plan,raw=records[names[1]]
assert raw['arms']=={} and not (base/'control_logits.npz').exists() and not (base/'candidate_logits.npz').exists()
log=(base/'parity.log').read_text();assert 'Prompt has 30907 tokens, fewer than requested 31500' in log
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(identity['snapshot_path'])
ids=tokenizer((ROOT/plan['prompt']).read_text(),return_tensors='pt')['input_ids']
assert ids.shape[1]==30907<plan['settings']['context']==31500
reports[names[1]]={'diagnostic_verified':True,'registered_parity_gate_passed':False,
 'reason':'pre-forward harness precondition: frozen prompt30907tokens shorter than requested31500',
 'tokenizer_count_independently_reproduced':int(ids.shape[1]),'arm_results':0,'persisted_logits_arrays':0,
 'quality_or_performance_outcome':None,'cleanup':'ExitStack cleanup is present in source but no successful-row hook/allocation counters were saved; do not claim runtime closure verification'}
out={'diagnostic_verified':True,'all_registered_stress_gates_passed':False,'source_attestation':checkpoint,
 'model_artifacts_rehashed':len(identity['artifact_sha256']),'cases':reports,
 'limits':['No new GPU/model experiments or favorable retries','Payload/logit equivalence is distinct from nativepath coverage','No physicalSTT/SOTA/timing or broad correctness claim'],
 'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for base,plan,raw in records.values() for p in [base/'plan.json',base/'run_parity.py',base/'parity.json',base/'parity.log']}}
output=ROOT/'results/audit_20261013/reviewer/verified_failed_exact_stress.json';assert not output.exists()
output.write_text(json.dumps(out,indent=2)+'\n');print('DIAGNOSTIC_VERIFIED_GATES_REJECTED')
