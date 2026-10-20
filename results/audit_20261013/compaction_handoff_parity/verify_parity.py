"""Read-only CPU verification of frozen full-model controller parity evidence."""
from pathlib import Path
import argparse,hashlib,json,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
from provenance import model_artifact_identity,runtime_identity

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
plan=json.loads((BASE/'plan.json').read_text());raw=json.loads((BASE/'parity.json').read_text())
archive_path=ROOT/plan['source_archive']
assert sha(archive_path)==plan['source_archive_sha256']
assert raw['plan_sha256']==sha(BASE/'plan.json')
assert plan['runner_sha256']==sha(BASE/'run_parity.py')
for name,value in plan['source_sha256'].items():assert sha(ROOT/name)==value,name
archive={}
with tarfile.open(archive_path,'r|gz') as tar:
    for member in tar:
        if member.isfile():
            key=member.name.removeprefix('./');stream=tar.extractfile(member)
            h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            archive[key]=h.hexdigest()
assert archive==plan['source_sha256'],'Archive manifest mismatch'
for name in ['prompt','teacher_stream']:assert sha(ROOT/plan[name])==plan[name+'_sha256']
assert model_artifact_identity(raw['model_identity']['snapshot_path'])==raw['model_identity']
assert runtime_identity()==raw['runtime']
settings=plan['settings']
assert settings['context'] in (8192,16384,31500) and settings['cold'] in (256,2048)
assert settings==dict(context=settings['context'],decode=50,hot=1024,cold=settings['cold'],fraction=.75,quota=32,recent=128,dtype='BF16',last_logits=True,seed=0)
a,b=[raw['arms'][name] for name in ['control','candidate']]
assert a['trace']==b['trace'];assert len(a['trace'])==51
fields={'hot_pos','cold_pos','cold_shadow','hot_k','hot_v','cold_k','cold_v','visible_k','visible_v','hot_cumulative','hot_age','cold_last_access','cold_peek_streak','cold_was_hot'}
for step in a['trace']:
    assert set(step)=={str(i) for i in range(32)}
    for row in step.values():
        assert set(row)==fields
        assert len(row['hot_pos'])==len(set(row['hot_pos']))<=settings['hot']
        assert len(row['cold_pos'])==len(set(row['cold_pos']))<=settings['cold']
        assert len(row['cold_shadow'])==len(row['cold_pos'])
        assert all(type(v) is bool for v in row['cold_shadow'])
        for key in fields-{'hot_pos','cold_pos','cold_shadow'}:
            assert len(row[key])==64;int(row[key],16)
assert a['predicted_token_ids']==b['predicted_token_ids'];assert len(a['predicted_token_ids'])==50
for arm in [a,b]:assert arm['forwards']==51 and arm['hooks_closed'] is True
assert b['cold_steps']==50 and b['cpu_append_calls']==50 and b['cpu_fallback_steps']==0
assert b['cold_boundary_fallback_calls']==1
assert b['bulk_payload_steps']+b['bulk_payload_fallback_steps']==50
assert raw['model_identity']==plan['model_identity']
allocation_events={}
for name,arm in raw['arms'].items():
 stats=arm['allocation_stats_after_reset']
 assert stats['live_allocations']==0 and stats['live_requested_bytes']==0
 live={};events=stats['events']
 for event in events:
  assert event['bytes']>0
  if event['kind']=='allocate':
   assert event['token'] not in live
   live[event['token']]=event['bytes']
  else:
   assert event['kind']=='release' and live.pop(event['token'])==event['bytes']
 assert not live
 allocation_events[name]=events
assert len(allocation_events['candidate'])==2
assert allocation_events['candidate'][0]['bytes']==2*32*8*(settings['cold']+64)*128*2
logit_shapes=[]
with np.load(BASE/'control_logits.npz') as left,np.load(BASE/'candidate_logits.npz') as right:
    assert set(left.files)==set(right.files)=={str(i) for i in range(51)}
    for i in range(51):
        x,y=left[str(i)],right[str(i)]
        assert x.shape==y.shape==(1,1,32000)
        assert x.dtype==y.dtype==np.dtype('float32')
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert x.tobytes()==y.tobytes()
        if i:assert int(x[0,-1].argmax())==a['predicted_token_ids'][i-1]
        logit_shapes.append(list(x.shape))
assert raw['passed'] is True and raw['trace_matches'] is True and raw['prediction_matches'] is True
assert raw['full_logits_bitwise_match'] is True and raw['max_logit_abs_difference']==0.0
out={'verified':True,'CPU_only':True,'settings':settings,'source_file_count':len(archive),'archive_equals_plan_equals_live_source':True,
     'model_artifact_count':len(raw['model_identity']['artifact_sha256']),'all_model_artifacts_rehashed':True,
     'runtime_identity_matches':True,'paired_trace_points':51,'layers_each':32,'full_vocab_logit_arrays':51,
     'logits_shape':[1,1,32000],'finite_and_bytes_exact':True,'predictions_recomputed_from_decode_logits':50,
     'candidate_native_cold_steps':50,'candidate_cold_boundary_initializer_fallbacks':1,'candidate_bulk_payload_steps':b['bulk_payload_steps'],'candidate_bulk_payload_fallback_steps':b['bulk_payload_fallback_steps'],'five_metadata_fields_exact_all_trace_points':True,'candidate_CPU_append_calls':50,'candidate_CPU_fallback_steps':0,
     'successful_run_hooks_closed':True,'exact_live_roots_zero_after_reset':True,'exact_allocation_events':allocation_events,
     'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [BASE/'plan.json',BASE/'run_parity.py',BASE/'parity.json',BASE/'control_logits.npz',BASE/'candidate_logits.npz',archive_path,Path(__file__)]},
     'limits':[f"one{settings['context']} reused input, coldbudget{settings['cold']} and50 teacherforced decode steps; not freegeneration/generalization",
              'trace includes KV/positions/shadows plus cumulative scores, ages, cold timestamps, hysteresis streak and prior-hot flags; native command/version/slot owners and broader generalization are not directly traced',
              'full payload copies/synchronizations are diagnostic and invalidate timing/traffic counters; no latency is saved',
              'CPU DRAM surrogate; not physical STT or SOTA/native payload-cost proof',
              'ExitStack guards successful acquired resources; partial constructor failures are not exercised']}
output=BASE/'verified_parity.json'
assert not output.exists(),'Refuse to overwrite independent proof'
output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
