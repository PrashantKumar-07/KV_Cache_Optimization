"""Recompute reserved-memory reduction and full-loop ABBA regression guards."""
import hashlib,json,tarfile,numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tarfile.open(B/'candidate_source.tar.gz') as archive:
 manifest={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
plan=json.loads((B/'abba_plan.json').read_text());assert sha(B/'verified_pilot.json')==plan['pilot_proof_sha256']
assert sha(B/'candidate_source.tar.gz')==plan['source_bundle_sha256']
results={}
for name,hot,cold in [('shared',991,1090),('default',1024,2048)]:
 raw=json.loads((B/f'abba_{name}.json').read_text());assert raw['passed'] and not raw.get('error')
 c=raw['config'];assert (c['hot'],c['cold'],c['decode'],c['blocks'],c['warmup'],c['context'])==(hot,cold,50,3,1,16384)
 assert raw['only_charged_prefill_cleanup_changed'] and all(manifest[p]==v for p,v in raw['source_sha256'].items())
 assert len(raw['rows'])==12
 reference=raw['rows'][0];arms={}
 for block in range(3):assert [r['arm'] for r in raw['rows'][block*4:block*4+4]]==plan['order']
 for row in raw['rows']:
  for key in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls'):
   assert row[key]==reference[key],(name,key)
  for key in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):assert row['trials'][0][key]==reference['trials'][0][key]
  m=row['whole_decode_resources'];assert m['root']['allocator_block_bytes']==m['root']['requested_storage_bytes']==(cold+64)*131072
  assert m['gpu_allocated_peak_bytes']==round(row['peak_memory_gb']*1e9)
  assert m['allocated_extra_plus_host_range_bytes']==m['gpu_allocated_peak_bytes']-m['constant_model_storage_bytes']+m['root']['allocator_block_bytes']
  assert m['reserved_extra_plus_host_range_bytes']==m['gpu_reserved_peak_bytes']-m['constant_model_storage_bytes']+m['root']['allocator_block_bytes']
 for arm in ('control','cleanup'):
  rows=[r for r in raw['rows'] if r['arm']==arm]
  steps=[v for r in rows for v in r['trials'][0]['step_ms']]
  assert len(steps)==300
  arms[arm]=dict(median_ms=float(np.median(steps)),p95_ms=float(np.percentile(steps,95)),
   ttft_ms=float(np.median([r['trials'][0]['ttft_ms'] for r in rows])),
   prefill_gpu_allocated_peak_gb=max(r['prefill_peak_memory_gb'] for r in rows),
   decode_gpu_reserved_peak_bytes=max(r['whole_decode_resources']['gpu_reserved_peak_bytes'] for r in rows),
   allocated_plus_host_peak_mib=max(r['whole_decode_resources']['allocated_extra_plus_host_range_bytes'] for r in rows)/2**20,
   reserved_plus_host_peak_mib=max(r['whole_decode_resources']['reserved_extra_plus_host_range_bytes'] for r in rows)/2**20,
   block_median_ms=[float(np.median([s for r in rows if r['block']==block for s in r['trials'][0]['step_ms']])) for block in range(3)])
 a,b=arms['control'],arms['cleanup']
 change={key:100*(b[key]/a[key]-1) for key in ('median_ms','p95_ms','ttft_ms','prefill_gpu_allocated_peak_gb')}
 reduction=100*(1-b['decode_gpu_reserved_peak_bytes']/a['decode_gpu_reserved_peak_bytes'])
 guards={key:change[key]<=limit for key,limit in [('median_ms',2),('p95_ms',2),('ttft_ms',10),('prefill_gpu_allocated_peak_gb',2)]}
 stats=raw['exact_allocation_stats'];assert stats['live_allocations']==stats['live_requested_bytes']==0
 events=stats['events'];assert len(events)==48
 assert {e['token'] for e in events if e['kind']=='allocate'}=={e['token'] for e in events if e['kind']=='release'}
 results[name]=dict(arms=arms,reserved_reduction_pct=reduction,change_pct=change,regression_guards=guards,
  advance_to_mandatory_integrity=reduction>=10 and all(guards.values()),active_512_mib_scope_pass=b['allocated_plus_host_peak_mib']<=512,
  reserved_512_mib_scope_pass=b['reserved_plus_host_peak_mib']<=512)
report=dict(verified=True,geometries=results,advance_to_mandatory_integrity=all(r['advance_to_mandatory_integrity'] for r in results.values()),
 candidate_ready_for_cli=False,new_speed_claim=False,model_wide_memory_claim=False,
 limitations='Memory architecture screen in the same serialized recorded timing pipeline. Decode GPU reserved peak only; active/prefill/request maximum remains separate. Complete driver/other host scopes excluded; reserved 512MiB target remains failed. No independent quality or physical STT claim.',
 evidence_sha256={p:sha(B/p) for p in ['abba_plan.json','verified_pilot.json','candidate_source.tar.gz','abba_shared.json','abba_default.json']},verifier_sha256=sha(Path(__file__)))
out=B/'verified_abba.json';assert not out.exists();out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
