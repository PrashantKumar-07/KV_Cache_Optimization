import hashlib,json,tarfile
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=p/'verified_report.json';assert not out.exists()
source=json.loads((p/'source_manifest.json').read_text());assert sha(p/'source.tar.gz')==source['bundle_sha256']
with tarfile.open(p/'source.tar.gz') as a:
 for path,h in source['source_sha256'].items():assert hashlib.sha256(a.extractfile(path).read()).hexdigest()==h
r=json.loads((p/'interleaved.json').read_text());assert r['passed'] and r['direct_interleaved'] and len(r['rows'])==9
for path,h in r['source_sha256'].items():assert h==source['source_sha256'][path]
rows={(x['mode'],x['request']):x for x in r['rows']};assert len(rows)==9
fields=('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters')
for req,steps in (('A',12),('B',24),('C',12)):
 for mode in ('gpu','cpu','interleaved_cpu'):
  x=rows[(mode,req)]
  assert len(x['predicted_token_ids'])==len(x['every_boundary_sha256'])==steps+1
  assert x['counters']['decode_steps']==steps
  for k in fields:assert x[k]==rows[('gpu',req)][k],(req,mode,k)
assert [(x['mode'],x['request']) for x in r['rows'][-3:]]==[('interleaved_cpu','A'),('interleaved_cpu','C'),('interleaved_cpu','B')]
a,b=[json.loads((p/f'cli_{name}.json').read_text()) for name in ('control','direct')]
for x in (a,b):
 assert x['config']['selector_sha256']==source['source_sha256']['experiments/evaluate_query_selection.py']
 assert x['config']['cpu_batched_scores'] and x['config']['strategy']=='lru' and x['config']['real_query_prefill']
assert not a['config']['cpu_direct_shared'] and b['config']['cpu_direct_shared']
assert b['config']['direct_backend_sha256']==source['source_sha256']['experiments/direct_shared_pool.py']
assert a['config']['dataset_sha256']==b['config']['dataset_sha256']
x,y=[v['results']['qasper']['tieredkv'] for v in (a,b)]
for k in ('score','n','sample_scores','cache_stats'):assert x[k]==y[k]
for k in ('sample_id','prediction','score','prompt_tokens','generated_tokens','final_resident_kv_payload_bytes','final_hot_kv_payload_bytes','final_cpu_kv_occupied_bytes','final_cpu_kv_reserved_bytes','submitted_h2d_bytes','submitted_d2h_bytes'):
 assert x['samples'][0][k]==y['samples'][0][k],k
flags=json.loads((p/'invalid_flags.json').read_text());assert flags['passed'] and len(flags['checks'])==2
report=dict(verified=True,source_bundle_sha256=source['bundle_sha256'],raw_sha256={n:sha(p/n) for n in ('interleaved.json','cli_control.json','cli_direct.json','invalid_flags.json')},
 direct_interleaved_ids_fullstate_51boundary_frames_and_counters_match_gpu_and_oldcpu=True,
 cli_matched_text_counts_scores_payload_counters=True,invalid_flags_rejected_before_model_load=True,
 limitations='One model/seed; serialized defaultstream fixedlength interleaving, EOSignored. CLI one reused Qasper item with nativeEOS but no tokenIDexport. No new quality/latency/servingthroughput/physicalSTT claim; earlier NarrativeQA NI remains unproven.')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
