"""Verify fresh-process allocator layout identity, snapshots and exact outputs."""
import hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tarfile.open(B/'candidate_source.tar.gz') as archive:
 manifest={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
cm=json.loads((B/'candidate_manifest.json').read_text());assert sha(B/'candidate_source.tar.gz')==cm['source_bundle_sha256']
assert manifest['experiments/audit_allocator_layout.py']==cm['driver_sha256']
results={}
for name,hot,cold in [('shared',991,1090),('default',1024,2048)]:
 data=[];pids=[]
 for layout,value in [('default','False'),('expandable','True')]:
  raw=json.loads((B/f'pilot_{name}_{layout}.json').read_text());assert raw['passed'] and not raw.get('error')
  assert (raw['config']['hot'],raw['config']['cold'],raw['config']['decode'],raw['config']['blocks'])==(hot,cold,12,0)
  assert len(raw['rows'])==2 and all(manifest[p]==v for p,v in raw['source_sha256'].items())
  data.append(raw)
  for row in raw['rows']:
   item=row['allocator_layout'];assert item['setting']=='expandable_segments:'+value and item['same_caller_in_both_rows']
   s=item['segments'];t=item['snapshot_totals']
   reserved=sum(x['total_size'] for x in s);allocated=sum(x['allocated_size'] for x in s);active=sum(x['active_size'] for x in s)
   trapped=sum(b['size'] for x in s if x['active_size']>0 for b in x['blocks'] if b['state']=='inactive')
   fully=sum(x['total_size'] for x in s if x['active_size']==0)
   assert (reserved,allocated,active,trapped,fully)==tuple(t[k] for k in ('reserved_bytes','allocated_bytes','active_bytes','inactive_in_live_segments_bytes','fully_inactive_segment_bytes'))
   assert active+trapped+fully==reserved
   assert bool(sum(x['is_expandable'] for x in s))==(value=='True')
   assert len(row['every_visible_state_sha256'])==13
   m=row['whole_decode_resources'];assert m['root']['requested_storage_bytes']==m['root']['allocator_block_bytes']==(cold+64)*131072
   assert m['gpu_reserved_peak_bytes']>=reserved and m['gpu_allocated_peak_bytes']>=allocated
  assert raw['exact_allocation_stats']['live_allocations']==raw['exact_allocation_stats']['live_requested_bytes']==0
 reference=data[0]['rows'][0]
 for raw in data:
  for row in raw['rows']:
   for f in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls'):assert row[f]==reference[f],(name,f)
   for f in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):assert row['trials'][0][f]==reference['trials'][0][f]
 arms={}
 for layout,raw in zip(('default','expandable'),data):
  arms[layout]=dict(gpu_reserved_peak_bytes=max(row['whole_decode_resources']['gpu_reserved_peak_bytes'] for row in raw['rows']),
    active_plus_host_peak_mib=max(row['whole_decode_resources']['allocated_extra_plus_host_range_bytes'] for row in raw['rows'])/2**20,
    reserved_plus_host_peak_mib=max(row['whole_decode_resources']['reserved_extra_plus_host_range_bytes'] for row in raw['rows'])/2**20,
    inactive_in_live_segments_peak_mib=max(row['allocator_layout']['snapshot_totals']['inactive_in_live_segments_bytes'] for row in raw['rows'])/2**20,
    fully_inactive_segment_peak_mib=max(row['allocator_layout']['snapshot_totals']['fully_inactive_segment_bytes'] for row in raw['rows'])/2**20)
 reduction=100*(1-arms['expandable']['gpu_reserved_peak_bytes']/arms['default']['gpu_reserved_peak_bytes'])
 results[name]=dict(arms=arms,reserved_reduction_pct=reduction,primary_pass=reduction>=10,
   advance_to_fresh_process_timing=reduction>=10,actual_resource_equality=False)
report=dict(verified=True,geometries=results,advance_geometries=[n for n,r in results.items() if r['primary_pass']],
 candidate_ready_for_cli=False,global_environment_changed=False,
 limitations='Two hash-instrumented runs per fresh layout process at12decode steps; no timing claim. Shared memory primary fails, usual primary passes narrowly. Both reserved-plus-host ceilings exceed512MiB; usual active ceiling also fails. No joint configuration/default, complete resource, model-wide active, quality or physical-STT claim.',
 evidence_sha256={p:sha(B/p) for p in ['plan.json','candidate_manifest.json','candidate_source.tar.gz']+[f'pilot_{n}_{s}.json' for n in ('shared','default') for s in ('default','expandable')]},verifier_sha256=sha(Path(__file__)))
out=B/'verified_pilot.json';assert not out.exists();out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
