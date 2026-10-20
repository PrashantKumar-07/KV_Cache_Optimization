"""Recompute three alternating fresh-process blocks; no shared-budget rescue."""
import hashlib,json,tarfile,numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cm=json.loads((B/'timing_manifest.json').read_text());assert sha(B/'timing_source.tar.gz')==cm['source_bundle_sha256']
with tarfile.open(B/'timing_source.tar.gz') as archive:
 manifest={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
assert manifest['experiments/time_allocator_layout.py']==cm['driver_sha256']
plan=json.loads((B/'timing_plan.json').read_text());assert sha(B/'verified_pilot.json')==plan['pilot_proof_sha256']
pilot=json.loads((B/'verified_pilot.json').read_text());assert pilot['advance_geometries']==['default']
rows=[];evidence=[];reference=None
for block in range(3):
 for position,layout in enumerate(plan['process_order']):
  filename=f'timing_b{block}_p{position}_{layout}.json';evidence.append(filename)
  raw=json.loads((B/filename).read_text());assert raw['passed'] and not raw.get('error')
  c=raw['config'];assert (c['hot'],c['cold'],c['decode'],c['warmup'],c['context'])==(1024,2048,50,1,16384)
  assert len(raw['rows'])==2 and all(manifest[p]==v for p,v in raw['source_sha256'].items())
  for row in raw['rows']:
   assert row['complete_state_hash_disabled_for_timing'] and row['every_visible_state_sha256']==[]
   a=row['allocator_layout'];assert a['setting']=='expandable_segments:'+str(layout=='expandable') and a['same_caller_in_both_rows']
   assert bool(sum(s['is_expandable'] for s in a['segments']))==(layout=='expandable')
   assert a['snapshot_totals']['reserved_bytes']==sum(s['total_size'] for s in a['segments'])
   assert row['whole_decode_resources']['root']['allocator_block_bytes']==276824064
   if reference is None:reference=row
   for key in ('state_sha256','history_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls'):
    assert row[key]==reference[key],(block,position,key)
   for key in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):assert row['trials'][0][key]==reference['trials'][0][key]
   assert len(row['trials'][0]['step_ms'])==50
   rows.append(dict(layout=layout,process_block=block,process_position=position,**row))
  stats=raw['exact_allocation_stats'];assert stats['live_allocations']==stats['live_requested_bytes']==0
  events=stats['events'];assert len(events)==8
  assert {e['token'] for e in events if e['kind']=='allocate'}=={e['token'] for e in events if e['kind']=='release'}
arms={}
for layout in ('default','expandable'):
 subset=[r for r in rows if r['layout']==layout]
 steps=[v for r in subset for v in r['trials'][0]['step_ms']];assert len(steps)==600
 arms[layout]=dict(median_ms=float(np.median(steps)),p95_ms=float(np.percentile(steps,95)),
  ttft_ms=float(np.median([r['trials'][0]['ttft_ms'] for r in subset])),
  prefill_allocated_peak_gb=max(r['prefill_peak_memory_gb'] for r in subset),
  gpu_reserved_peak_bytes=max(r['whole_decode_resources']['gpu_reserved_peak_bytes'] for r in subset),
  active_plus_host_peak_mib=max(r['whole_decode_resources']['allocated_extra_plus_host_range_bytes'] for r in subset)/2**20,
  reserved_plus_host_peak_mib=max(r['whole_decode_resources']['reserved_extra_plus_host_range_bytes'] for r in subset)/2**20,
  inactive_in_live_segments_peak_mib=max(r['allocator_layout']['snapshot_totals']['inactive_in_live_segments_bytes'] for r in subset)/2**20,
  block_median_ms=[float(np.median([v for r in subset if r['process_block']==b for v in r['trials'][0]['step_ms']])) for b in range(3)])
a,b=arms['default'],arms['expandable']
change={k:100*(b[k]/a[k]-1) for k in ('median_ms','p95_ms','ttft_ms','prefill_allocated_peak_gb')}
reduction=100*(1-b['gpu_reserved_peak_bytes']/a['gpu_reserved_peak_bytes'])
guards={k:change[k]<=limit for k,limit in [('median_ms',2),('p95_ms',2),('ttft_ms',10),('prefill_allocated_peak_gb',2)]}
report=dict(verified=True,geometry='default',fresh_processes=12,measured_tokens_per_layout=600,arms=arms,
 reserved_reduction_pct=reduction,change_pct=change,regression_guards=guards,
 advance_to_mandatory_integrity=reduction>=10 and all(guards.values()),shared_pilot_rejected=True,
 candidate_ready_for_cli=False,actual_resource_equality=False,
 limitations='Twelve alternating fresh processes, two identical cleanup runs each, same legacy history/synchronization instrumentation. No full hash observer in timing; final complete state and full token/counter/payload histories exact. Usual geometry only; shared pilot remains rejected. Active and reserved-plus-host512MiB gates remain failed; no model-wide active, new speed, independent quality, serving or physical-STT claim.',
 evidence_sha256={p:sha(B/p) for p in evidence+['verified_pilot.json','timing_plan.json','timing_manifest.json','timing_source.tar.gz']},verifier_sha256=sha(Path(__file__)))
out=B/'verified_timing.json';assert not out.exists();out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
