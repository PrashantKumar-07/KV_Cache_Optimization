"""Independent source/geometry/equation check of shared resource ceilings."""
import hashlib,json,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent
raw=json.loads((P/'result.json').read_text());plan=json.loads((P/'plan.json').read_text());manifest=json.loads((P/'source_manifest.json').read_text())
assert raw['passed'] and raw['settings']==plan['settings']
assert hashlib.sha256((P/'source.tar.gz').read_bytes()).hexdigest()==manifest['bundle_sha256']
with tarfile.open(P/'source.tar.gz') as archive:
 for name,sha in raw['source_sha256'].items():
  assert sha==manifest['source_sha256'][name]==hashlib.sha256(archive.extractfile(name).read()).hexdigest()
assert raw['prompt_sha256']==plan['prompt_sha256'] and raw['teacher_source_sha256']==plan['teacher_sha256']
assert len(raw['input_token_ids'])==16384 and len(raw['teacher_token_ids'])==50
summary=[]
for arm in raw['rows']:
 rows=arm['boundaries'];assert len(rows)==51 and [r['step'] for r in rows]==list(range(51))
 totals=[];gpu=[];host=[];widths=[]
 for r in rows:
  assert len(r['layers'])==32
  total=sum(r['distinct_owned_kv_storage_bytes'].values());totals.append(total)
  g=r['distinct_owned_kv_storage_bytes'].get('cuda:0',0);h=r['distinct_owned_kv_storage_bytes'].get('cpu',0);gpu.append(g);host.append(h)
  assert g+h==total and total<=plan['owned_kv_ceiling_bytes']
  layers=r['layers'];visible=sum(x['visible_rows']*x['stored_visible_heads']*128*2*2 for x in layers)
  assert visible==r['visible_kv_tensor_payload_bytes']
  if arm['arm']=='snap':
   assert all(x['visible_rows']==974+r['step'] and x['stored_visible_heads']==32 for x in layers)
   assert visible==g and h==0
  else:
   assert all(x['hot_rows']==991 and x['visible_rows']==1023 and x['stored_visible_heads']==8 for x in layers)
   hot=sum(x['hot_payload_bytes'] for x in layers)
   if arm['arm']=='gpu':
    cold=sum(x['cold_reserved_payload_bytes'] for x in layers);assert visible+hot+cold==g and h==0
   else:
    assert h==2082*131072 and visible+hot==g
  if r['step']:
   current=r['forward_attended_rows_by_layer'];assert len(current)==32
   assert max(current)<=plan['forward_attended_row_ceiling'];widths+=current
 assert len(arm['predicted_token_ids'])==50
 summary.append(dict(arm=arm['arm'],total_owned_kv_range_bytes=[min(totals),max(totals)],gpu_owned_kv_range_bytes=[min(gpu),max(gpu)],cpu_owned_kv_range_bytes=[min(host),max(host)],decode_attended_rows_range=[min(widths),max(widths)],final_distinct_owned_kv_storage_bytes=rows[-1]['distinct_owned_kv_storage_bytes'],diagnostic_model_peak_gpu_allocated_bytes=arm['diagnostic_peak_gpu_allocated_bytes']))
output=dict(source_identity_and_boundary_storage_equations_verified=True,shared_ceiling_passed=True,actual_usage_equal=False,arms=summary,boundaries=153,layer_decode_width_observations=4800,raw_sha256=hashlib.sha256((P/'result.json').read_bytes()).hexdigest(),source_bundle_sha256=manifest['bundle_sha256'],limitations=raw['limitations'])
dest=P/'verified_report.json';assert not dest.exists();dest.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
