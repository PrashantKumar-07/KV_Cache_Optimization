"""Independent CPU-only verifier; never imports/runs the GPU benchmark driver."""
from pathlib import Path
import datetime, hashlib, importlib.metadata, json, math, statistics, sys, tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; BASE=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def quantile(values,q):
 a=sorted(values);r=(len(a)-1)*q;i=int(r);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(r-i)
def near(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),(a,b)
def stats(trials):
 steps=[v for t in trials for v in t['step_ms']]
 return dict(measured_decode_forwards=len(steps),median_itl_ms=statistics.median(steps),p95_itl_ms=quantile(steps,.95),median_ttft_ms=statistics.median([t['ttft_ms'] for t in trials]))

def main():
 output=BASE/'independently_verified_benchmark.json';assert not output.exists(),'Preserve prior evidence'
 p=load(BASE/'plan.json');r=load(BASE/'benchmark.json')
 assert r.get('execution_complete') is True and 'error' not in r,'Benchmark not successfully terminal; do not verify partial outcomes'
 assert r['plan_sha256']==sha(BASE/'plan.json') and sha(BASE/'run_benchmark.py')==p['runner_sha256']
 assert r['runtime']==p['runtime']
 for name,version in r['runtime']['packages'].items():assert importlib.metadata.version(name)==version
 assert sha(ROOT/p['source_archive'])==p['source_archive_sha256']
 archive={}
 with tarfile.open(ROOT/p['source_archive'],'r|gz') as tf:
  for m in tf:
   if m.isfile():archive[m.name.removeprefix('./')]=hashlib.sha256(tf.extractfile(m).read()).hexdigest()
 assert archive==p['source_sha256'] and len(archive)==5918
 assert all(sha(ROOT/n)==h for n,h in archive.items()),'Live source changed since measurement'
 prior=load(ROOT/'results/audit_20261013/reusable_root_parity/plan.json')['source_sha256']
 changed={n:dict(prior=h,current=archive.get(n)) for n,h in prior.items() if archive.get(n)!=h}
 assert changed==p['prior_native_proof_source_bridge']['changed_prior_files']
 assert sorted(set(archive)-set(prior))==sorted(p['prior_native_proof_source_bridge']['current_added_files'])
 for n,h in p['prerequisite_sha256'].items():assert sha(ROOT/n)==h,n
 for n in ('prompt','teacher_stream','quiet_helper'):assert sha(ROOT/p[n])==p[n+'_sha256']
 for n,h in p['model_identity']['artifact_sha256'].items():assert sha(Path(p['model_identity']['snapshot_path'])/n)==h,n
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(p['model_identity']['snapshot_path'],local_files_only=True)
 ids=tok((ROOT/p['prompt']).read_text(),return_tensors='np')['input_ids']
 assert ids.shape==(1,p['full_prompt_tokens'])
 for ctx in p['contexts']:assert hashlib.sha256(ids[:,:ctx].astype(np.int64,copy=False).tobytes()).hexdigest()==p['input_ids_sha256'][str(ctx)]
 assert r['input_ids_sha256']==p['input_ids_sha256']
 # Confirm the changed harness did not rerun a prior measured failure.
 old=load(ROOT/p['harness_correction']['prior_attempt']);diag=load(ROOT/p['harness_correction']['diagnostic'])
 assert old['execution_complete'] is False and not old['rows'] and diag['execution_complete'] is False and not diag['rows']
 framework=Path(__import__('transformers').__file__).parent/'utils/output_capturing.py'
 assert sha(framework)==p['harness_correction']['framework_source_sha256']
 expected_hook='transformers.utils.output_capturing:install_output_capuring_hook.<locals>.output_capturing_hook'
 expected_names={name for i in range(32) for name in (f'model.layers.{i}',f'model.layers.{i}.self_attn')}
 assert set(r['framework_output_hooks'])==expected_names
 assert all(v==[expected_hook] for v in r['framework_output_hooks'].values())
 schedule=[(True,16384,a) for a in ('snap','native')]+[(False,c,a) for c in p['contexts'] for a in p['order']]
 assert len(r['diagnostics'])==2 and len(r['rows'])==36
 rows=r['diagnostics']+r['rows'];assert len(rows)==len(schedule)
 expected_quiet=[('before_model_load',None,None)]
 for diagnostic,ctx,arm in schedule:expected_quiet.extend([('before_diagnostic' if diagnostic else 'before_row',ctx,arm),('after_diagnostic' if diagnostic else 'after_row',ctx,arm)])
 assert len(r['quiet_observations'])==p['expected_quiet_observations']==77
 for q,expected in zip(r['quiet_observations'],expected_quiet):
  assert (q['stage'],q.get('context'),q.get('arm'))==expected
  assert q['quiet'] is True and not q['unexpected_compute_processes']
  assert q['gpu'].startswith('0, GPU-5ae5c9c4-c441-9622-9599-9f6d69aa81a8,')
  assert all(int(line.split(',')[1].strip())==r['pid'] for line in q['compute_processes'])
 reference={};pool_ptr=None;leases=0;allocation_event=None;native_counts=[]
 for i,(row,(diagnostic,ctx,arm)) in enumerate(zip(rows,schedule)):
  assert (row['index'],row['context'],row['arm'])==(i,ctx,arm)
  assert row['hooks_closed'] is True and row['lease_closed'] is True
  result=row['result'];assert result['timing_scope']==('diagnostic_invalid_timing' if diagnostic else 'end_to_end_model_policy_token')
  trials=result['trials'];assert len(trials)==(1 if diagnostic else 3)
  assert [t['warmup'] for t in trials]==([False] if diagnostic else [True,False,False])
  for t in trials:
   assert len(t['step_ms'])==len(t['forward_ms'])==len(t['predicted_token_ids'])==50
   assert all(math.isfinite(v) and v>0 for v in t['step_ms']+t['forward_ms']+[t['ttft_ms'],t['prefill_forward_ms']])
   assert t['all_forward_logits_finite'] is True
   assert all(0<=v<32000 for v in [t['prefill_prediction']]+t['predicted_token_ids'])
   trajectory=[t['prefill_prediction']]+t['predicted_token_ids'];key=(ctx,arm)
   reference.setdefault(key,trajectory);assert trajectory==reference[key]
   if str(ctx) in p['reference_predictions'] and arm in p['reference_predictions'][str(ctx)]:assert t['predicted_token_ids']==p['reference_predictions'][str(ctx)][arm]
   assert t['final_visible_rows']==[954 if arm=='snap' else 1056]*32
   if arm=='native':
    leases+=1;c=t['counts'];native_counts.append(c)
    assert c['cold_boundary_steps']==c['cpu_append_calls']==50 and c['cpu_fallback_steps']==0
    assert c['cold_boundary_fallback_calls']==1 and c['bulk_payload_steps']+c['bulk_payload_fallback_steps']==50
    assert c['bulk_payload_steps']==50 and c['bulk_payload_fallback_steps']==0
    info=t['actual_pinned_pool'];assert info['requested_storage_bytes']==info['allocator_block_bytes']==276824064
    assert info['allocated'] is True and info['public_cuda_memory_type']==1
    if pool_ptr is None:pool_ptr=t['pool_pointer']
    assert t['pool_pointer']==pool_ptr
  assert row['owner_leases']==leases
  s=row['exact_allocation_stats_after_cleanup'];allocated=i>0
  assert s['live_allocations']==int(allocated) and s['live_requested_bytes']==276824064*int(allocated)
  if not allocated:
   assert s['events']==[] and row['experiment_retained_idle_root'] is None and row['retained_idle_pointer'] is None
  else:
   assert len(s['events'])==1 and s['events'][0]['kind']=='allocate' and s['events'][0]['bytes']==276824064
   if allocation_event is None:allocation_event=s['events'][0]
   assert s['events']==[allocation_event] and row['retained_idle_pointer']==pool_ptr
   info=row['experiment_retained_idle_root'];assert info['requested_storage_bytes']==info['allocator_block_bytes']==276824064 and info['allocated'] is True
 assert leases==p['expected_native_leases']==55 and r['owner_closed'] is True
 final=r['final_exact_allocation_stats'];assert final['live_allocations']==final['live_requested_bytes']==0
 assert len(final['events'])==2 and final['events'][0]==allocation_event
 assert final['events'][1]==dict(kind='release',token=allocation_event['token'],bytes=276824064)
 # Independently derive the existing-reference predictions, including prefill.
 for ctx,arm,path in [(16384,'snap','results/audit_20261013/snap_projection_reuse_gpu/reused_logits.npz'),(16384,'native','results/audit_20261013/reusable_root_parity/16384_candidate_logits.npz'),(8192,'native','results/audit_20261013/reusable_root_parity/8192_candidate_logits.npz')]:
  with np.load(ROOT/path) as arrays:
   assert set(arrays.files)=={str(j) for j in range(51)}
   pred=[]
   for j in range(51):
    a=arrays[str(j)];assert a.shape==(1,1,32000) and np.isfinite(a).all();pred.append(int(a[0,-1].argmax()))
   assert pred==reference[(ctx,arm)]
 diagnostics={}
 for row in r['diagnostics']:
  arm=row['arm'];frames=row['result']['boundaries'];assert len(frames)==51
  for j,f in enumerate(frames):
   assert len(f['layers'])==32
   expected_width=904+j if arm=='snap' else 1056;heads=32 if arm=='snap' else 8
   assert all(l['layer']==i and l['visible_rows']==expected_width and l['visible_heads']==heads for i,l in enumerate(f['layers']))
   assert f['visible_tensor_payload_bytes']==32*2*heads*expected_width*128*2
   if arm=='snap':assert f['deduplicated_named_KV_storage_bytes']=={'cuda:0':f['visible_tensor_payload_bytes']} and f['cpu_canonical_metadata_bytes']==0
   else:
    assert f['deduplicated_named_KV_storage_bytes']=={'cuda:0':32*2*8*1057*128*2,'cpu':276824064}
    assert all(l['hot_rows']==1024 and 1<=l['cold_live_rows']<=l['cold_total_rows']<=2048 for l in f['layers'])
    assert f['cpu_canonical_metadata_bytes']==row['result']['trials'][0]['cpu_canonical_arrays_bytes']
  diagnostics[arm]={'boundaries':51,'initial_visible_width':frames[0]['layers'][0]['visible_rows'],'final_visible_width':frames[-1]['layers'][0]['visible_rows'],'visible_heads':heads,'initial_named_storage':frames[0]['deduplicated_named_KV_storage_bytes'],'final_named_storage':frames[-1]['deduplicated_named_KV_storage_bytes'],'initial_gpu_allocated_bytes':frames[0]['gpu_allocated_bytes'],'final_gpu_allocated_bytes':frames[-1]['gpu_allocated_bytes'],'postcommit_source_order_reviewed':True}
 per_context={};memory={};static=r['constant_model_storage_bytes']
 for ctx in p['contexts']:
  ctxrows=[row for row in r['rows'] if row['context']==ctx];assert [row['arm'] for row in ctxrows]==p['order']
  arms={};memory[str(ctx)]={}
  for arm in ('snap','native'):
   trials=[t for row in ctxrows if row['arm']==arm for t in row['result']['trials'] if not t['warmup']]
   assert len(trials)==12;arms[arm]=stats(trials);assert arms[arm]['measured_decode_forwards']==600
   terms=[]
   for t in trials:
    d={'decode_GPU_excluding_static_model':t['decode_peak_gpu_allocated_bytes']-static};assert d['decode_GPU_excluding_static_model']>=0
    if arm=='native':
     c=t['counts'];d.update(observed_main_pinned_allocator_block=t['actual_pinned_pool']['allocator_block_bytes'],canonical_CPU_arrays=t['cpu_canonical_arrays_bytes'],CPU_streak_array_peak=c['controller_cpu_streak_array_peak'],CPU_address_array_peak=c['controller_address_array_peak'],registered_two_times_cumulative_descriptors=2*(c['bulk_descriptor_h2d_bytes']+c['cold_boundary_descriptor_h2d_bytes']))
    terms.append(dict(terms_bytes=d,scoped_bytes=sum(d.values())))
   memory[str(ctx)][arm]=terms
   arms[arm]['max_scoped_bytes']=max(t['scoped_bytes'] for t in terms)
   for stage in ('prefill','decode'):
    for kind in ('allocated','reserved'):
     field=f'{stage}_peak_gpu_{kind}_bytes';arms[arm]['max_'+field]=max(t[field] for t in trials)
   for metric,value in arms[arm].items():near(value,r['summary']['per_context'][str(ctx)]['arms'][arm][metric])
  ratios={m:arms['native'][m]/arms['snap'][m] for m in p['timing_ratio_guards']}
  guards={m:ratios[m]<=limit for m,limit in p['timing_ratio_guards'].items()}
  guards.update({arm+'_scoped_budget':arms[arm]['max_scoped_bytes']<=p['scoped_budget_bytes'] for arm in arms})
  assert guards==r['summary']['per_context'][str(ctx)]['guards']
  blocks=[]
  for b in range(3):
   med={a:statistics.median([x for row in ctxrows[b*4:(b+1)*4] if row['arm']==a for t in row['result']['trials'] if not t['warmup'] for x in t['step_ms']]) for a in arms}
   ratio=med['native']/med['snap'];near(ratio,r['summary']['per_context'][str(ctx)]['ABBA_blocks'][b]['native_snap_ratio']);blocks.append(dict(block=b,median_itl_ms=med,native_snap_ratio=ratio))
  per_context[str(ctx)]=dict(arms=arms,native_snap_ratios=ratios,guards=guards,ABBA_blocks=blocks,advancement_passed=all(guards.values()))
 assert memory==r['summary']['scoped_memory_terms']
 pooled={a:stats([t for row in r['rows'] if row['arm']==a for t in row['result']['trials'] if not t['warmup']]) for a in ('snap','native')}
 for a in pooled:
  assert pooled[a]['measured_decode_forwards']==1800
  for m,v in pooled[a].items():near(v,r['summary']['pooled']['arms'][a][m])
 ratios={m:pooled['native'][m]/pooled['snap'][m] for m in p['timing_ratio_guards']};guards={m:ratios[m]<=lim for m,lim in p['timing_ratio_guards'].items()}
 assert guards==r['summary']['pooled']['guards']
 passed=all(v['advancement_passed'] for v in per_context.values()) and all(guards.values());assert passed==r['summary']['advancement_passed']
 proof={'independently_verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'source_files_archive_live_plan_exact':len(archive),'model_artifacts_rehashed':len(p['model_identity']['artifact_sha256']),'runtime_exact':True,'prior_failed_runs_zero_timed_rows_preserved':True,'framework_hooks_exact_module_names_and_qualnames':64,'quiet_checks':77,'native_leases':leases,'one_root_pointer_and_allocate_release_pair':True,'idle_pinned_bytes':276824064,'final_live_roots':0,'every_forward_finite_flag_and_prediction_repeat_checked':True,'prior_full_vocab_derived_reference_predictions_including_prefill_match':['snap16384','native16384','native8192'],'native_all55requests_50cold50bulk50cpu_steps':True,'diagnostics':diagnostics,'per_context':per_context,'pooled':{'arms':pooled,'native_snap_ratios':ratios,'guards':guards},'scoped_memory_terms':memory,'advancement_passed':passed,'decision':'advance only registered timing/resource screen' if passed else 'REJECT advancement; preserve failed guards; no favorable rerun','limits':p['limits']+['No stored fullvocabulary arrays for this timingrun; finite flags reviewed against runner, predictions checked. FullKV/logit parity missing for snap8K and both30K as preregistered.','Framework guard records qualified function identities and exact per-module hook counts, not Python object identity at every boundary.','No statistical uncertainty interval on timing ratios preregistered; three correlated contextprefixes are not corpusgeneralization.'],'evidence_sha256':{str(path.relative_to(ROOT)):sha(path) for path in [BASE/'plan.json',BASE/'run_benchmark.py',BASE/'benchmark.json',Path(__file__)]}}
 output.write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps({k:proof[k] for k in ('independently_verified','per_context','pooled','advancement_passed','decision')},indent=2))
if __name__=='__main__':main()
