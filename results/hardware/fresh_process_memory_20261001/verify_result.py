import hashlib
import json
from pathlib import Path
p=Path(__file__).resolve().parent
raws=[json.loads((p/f'{arm}.json').read_text()) for arm in ('gpu','cpu')]
a,b=raws
assert all(r['passed'] for r in raws) and a['pid']!=b['pid']
for r in raws:
 for path,h in r['source_sha256'].items():assert hashlib.sha256((p/Path(path).name).read_bytes()).hexdigest()==h
 assert len(r['boundaries'])==52 and len(r['predicted_token_ids'])==50
 assert r['owned_pinned_after_cleanup_bytes']==0
 assert r['rss_samples']
 for x in r['boundaries']+r['rss_samples']+r['cleanup']:
  assert x['VmHWM_bytes']>=x['VmRSS_bytes']>0
 for x in r['boundaries']+r['cleanup']:
  assert x['gpu_reserved_bytes']>=x['gpu_allocated_bytes']>=0
  assert x['gpu_peak_reserved_bytes']>=x['gpu_reserved_bytes']
  assert x['gpu_peak_allocated_bytes']>=x['gpu_allocated_bytes']
 assert r['measurement_final']['gpu_peak_allocated_bytes']>=max(x['gpu_allocated_bytes'] for x in r['boundaries'])
 current=peak=0
 for event in r['pinned_root_events']:
  current+=(1 if event['kind']=='allocate' else -1)*event['bytes']
  assert current==event['owned_bytes'] and current>=0
  peak=max(peak,current)
 assert current==0 and peak==r['owned_pinned_peak_bytes']
for k in ('runtime','prompt_sha256','teacher_source_sha256','teacher_token_ids','input_ids_sha256','source_sha256','predicted_token_ids','state_sha256','counters'):
 assert a[k]==b[k],k
assert a['owned_pinned_current_bytes']==a['owned_pinned_peak_bytes']==a['reserved_host_bytes']==0
assert b['owned_pinned_current_bytes']==b['reserved_host_bytes']==402784256
assert b['owned_pinned_peak_bytes']==805568512
assert sum(x['kind']=='allocate' for x in b['pinned_root_events'])==33
assert not a['host_allocator_stats_available'] and not b['host_allocator_stats_available']
summary={}
for name,r in zip(('gpu','cpu'),raws):
 summary[name]=dict(gpu_peak_allocated_bytes=r['measurement_final']['gpu_peak_allocated_bytes'],
  gpu_peak_reserved_bytes=r['measurement_final']['gpu_peak_reserved_bytes'],
  loaded_model_rss_bytes=r['boundaries'][0]['VmRSS_bytes'],
  generation_sampled_rss_peak_bytes=max(x['VmRSS_bytes'] for x in r['rss_samples']),
  generation_lifetime_process_highwater_bytes=r['measurement_final']['VmHWM_bytes'],
  gpu_final_allocated_bytes=r['measurement_final']['gpu_allocated_bytes'],
  owned_pinned_peak_bytes=r['owned_pinned_peak_bytes'],owned_pinned_final_bytes=r['owned_pinned_current_bytes'],
  occupied_host_bytes=r['occupied_host_bytes'],cleanup=r['cleanup'])
gain=100*(1-summary['cpu']['gpu_peak_allocated_bytes']/summary['gpu']['gpu_peak_allocated_bytes'])
report=dict(verified=True,raw_sha256={name:hashlib.sha256((p/f'{name}.json').read_bytes()).hexdigest() for name in ('gpu','cpu')},
 source_sha256=a['source_sha256'],fresh_distinct_processes_inputs_ids_fullstate_counters_equal=True,
 arms=summary,gpu_modelwide_peak_reduction_percent=gain,ten_percent_peak_threshold_met=gain>=10,
 promotion_eligible=False,limitations=b['limitations'])
(p/'verified_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
