"""Independent integer reconstruction; no imports from accounting helpers."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,tarfile,math,statistics,sys
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 arm=sys.argv[1] if len(sys.argv)>1 else 'tiered_hot1024'
 assert arm in ('tiered_hot1024','tiered_hot992')
 suffix='_hot992' if arm=='tiered_hot992' else ''
 report_path=BASE/f'whole_request_writes{suffix}.json'
 report=json.loads(report_path.read_text())
 proof_path=ROOT/'results/full_evaluation_20261015/verified_results.json'
 proof=json.loads(proof_path.read_text())
 assert proof['verified'] and proof['execution_complete']
 freeze=json.loads((ROOT/'results/full_evaluation_20261015/freeze.json').read_text())
 archive=ROOT/'results/full_evaluation_20261015/source.tar.gz'
 assert sha(archive)==freeze['source_archive_sha256']
 frozen={}
 with tarfile.open(archive) as t:
  for name in ('src/tiered_kv_cache.py','src/metrics.py','experiments/longbench_eval.py'):
   data=t.extractfile(name).read()
   frozen[name]={'archive_member_sha256':hashlib.sha256(data).hexdigest(),'current_source_matches_frozen':sha(ROOT/name)==hashlib.sha256(data).hexdigest()}
   assert frozen[name]['current_source_matches_frozen']
 raw={k:v for k,v in proof['record_sha256'].items() if k.endswith('_'+arm+'.json')}
 assert len(raw)==1474 and report['raw_record_sha256']==raw
 published={(r['task'],r['index']):r for r in report['rows']}
 totals=defaultdict(lambda:[0,0,0,0,0])
 counts=defaultdict(int)
 for name,digest in raw.items():
  path=ROOT/name;assert sha(path)==digest
  r=json.loads(path.read_text());assert r['execution_complete'] and r['arm']==arm
  layers=r['geometry_after_prefill_before_first_decode']['layers']
  assert len(layers)==32 and {l['layer'] for l in layers}==set(range(32))
  initial_rows=initial_bytes=0
  for layer in layers:
   payloads=layer['tier_payloads'];k=payloads['stt_k'];v=payloads['stt_v']
   h,n,d=k['shape'];assert (h,d)==(8,128) and v['shape']==[h,n,d]
   assert k['dtype']==v['dtype']=='torch.bfloat16'
   assert k['logical_bytes']==v['logical_bytes']==h*n*d*2
   assert k['backing_storage_bytes']==k['logical_bytes'] and v['backing_storage_bytes']==v['logical_bytes']
   assert layer['cold_live_rows']==n and layer['cold_shadow_rows']==0
   initial_rows+=n;initial_bytes+=k['logical_bytes']+v['logical_bytes']
  assert initial_bytes==initial_rows*4096
  s=r['cache_stats'];demotions=s['demoted'];skipped=s['writes_saved'];paid=s['paid_writes']
  assert all(type(x) is int and x>=0 for x in (demotions,skipped,paid))
  assert demotions-skipped==paid
  a=published[r['task'],r['physical_file_row_index']]
  expected={'initial_cold_payload_row_writes':initial_rows,'decoder_paid_writes':paid,'decoder_avoided_writes':skipped,'whole_request_paid_payload_row_writes':initial_rows+demotions-skipped,'potential_row_writes_without_backup_reuse':initial_rows+demotions,'whole_request_paid_payload_bytes':initial_bytes+paid*4096,'whole_request_avoided_percent':100*skipped/(initial_rows+demotions) if initial_rows+demotions else None,'decoder_only_avoided_percent':100*skipped/demotions if demotions else None}
  assert a['sample_id']==r['sample_id'] and a['generated_tokens']==r['generated_tokens']
  for key,value in expected.items():
   assert a[key]==value,(name,key,a[key],value)
  for task in (r['task'],'all_tasks'):
   counts[task]+=1
   for i,value in enumerate((initial_rows,demotions,skipped,paid,r['generated_tokens'])):totals[task][i]+=value
 summaries={}
 for task,(initial,demoted,skipped,paid,tokens) in totals.items():
  sums={'initial_cold_payload_row_writes':initial,'decoder_paid_writes':paid,'decoder_avoided_writes':skipped,'whole_request_paid_payload_row_writes':initial+paid,'potential_row_writes_without_backup_reuse':initial+demoted,'whole_request_paid_payload_bytes':(initial+paid)*4096,'whole_request_avoided_percent':100*skipped/(initial+demoted),'decoder_only_avoided_percent':100*skipped/demoted}
  assert sums==report['summaries'][task],task
  summaries[task]={'records':counts[task],'generated_tokens':tokens,**sums}
 tasks=sorted(t for t in totals if t!='all_tasks')
 macro=statistics.mean(summaries[t]['whole_request_avoided_percent'] for t in tasks)
 macro_decode=statistics.mean(summaries[t]['decoder_only_avoided_percent'] for t in tasks)
 qa=[t for t in tasks if t!='gov_report']
 qi=sum(totals[t][0] for t in qa);qd=sum(totals[t][1] for t in qa);qs=sum(totals[t][2] for t in qa)
 output={'verified':True,'arm':arm,'records':1474,'independent_of_accounting_helpers':True,'reviewer_script_sha256':sha(Path(__file__)),'reviewed_report_sha256':sha(report_path),'quality_proof_sha256':sha(proof_path),'source_archive_sha256':sha(archive),'frozen_source_members':frozen,'independently_recomputed_summaries':summaries,'equal_task_macro_whole_request_avoided_percent':macro,'equal_task_macro_decode_only_avoided_percent':macro_decode,'five_qa_pooled_whole_request_avoided_percent':100*qs/(qi+qd),'gov_report_share_of_avoided_rows_percent':100*totals['gov_report'][2]/totals['all_tasks'][2],'gov_report_share_of_counterfactual_rows_percent':100*(totals['gov_report'][0]+totals['gov_report'][1])/(totals['all_tasks'][0]+totals['all_tasks'][1]),'source_review':{'initial':'Frozen src/tiered_kv_cache.py initial_bifurcation selects up to int(cold_capacity*fraction) remaining prompt positions, clones selected cold K and V only. GPU hot clones, dropped prompt positions and metadata are excluded. Raw cold tensors have exact logical/backing bytes and no initial shadow rows.','decode':'Each selected demotion increments demoted. A matching shadow increments writes_saved and avoids append_stt_pair. Nonmatching victims are appended as K/V payload. Thus decoder paid=demoted-saved counts logical newly appended payload rows, not all allocator or tensor-copy traffic.','geometry':'One layer-token row means K plus V over eight KV heads,128 elements/head,two bytes/BF16:2*8*128*2=4096 bytes. Sum over32 layers; do not multiply layer totals by32 again.'},'interpretation':{'legitimate_result':'The trace-based logical cold-payload accounting confirms write avoidance, including initial retained cold placement. The smaller whole-request percentage follows directly from adding actual initial cold placement to both measured and counterfactual counts.','counterfactual':'Fixed measured inclusive-policy trace with every demotion charged. This is not an actual exclusive-policy run: different backup retention/capacity can alter victims, retrieval, generation and quality. No causal exclusive-policy superiority is established.','denominator':'Whole-request here means only declared initial retained cold K/V placement plus logical decode-demotion payload writes; not all model prefill writes, cold metadata, compaction/reallocation copies, hardware transfers or physical device programming.','simulation':'Raw cold payloads are GPU tensors in this simulation. No physical STT write, endurance, lifetime or energy benefit can be inferred from these counters.','aggregation':'Pooled write fraction is workload/trace-volume weighted and strongly influenced by long GovReport generations. Equal-task macro and per-task values must accompany it for balanced task interpretation.'}}
 (BASE/f'whole_request_writes{suffix}_independent_proof.json').write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps({'verified':True,'records':1474,'pooled_whole':summaries['all_tasks']['whole_request_avoided_percent'],'macro_whole':macro,'macro_decode':macro_decode,'five_qa_pooled_whole':output['five_qa_pooled_whole_request_avoided_percent'],'gov_share_avoided':output['gov_report_share_of_avoided_rows_percent']},indent=2))
if __name__=='__main__':main()
