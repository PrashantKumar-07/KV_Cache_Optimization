"""Verify the two short preservation/resource pilots without a speed claim."""
import hashlib,json,tarfile,numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tarfile.open(B/'candidate_source.tar.gz') as a:
 manifest={m.name:hashlib.sha256(a.extractfile(m).read()).hexdigest() for m in a.getmembers() if m.isfile()}
cm=json.loads((B/'candidate_manifest.json').read_text());assert sha(B/'candidate_source.tar.gz')==cm['source_bundle_sha256']
assert manifest['experiments/screen_prefill_reclaim.py']==cm['driver_sha256']
results={}
for name,hot,cold in [('shared',991,1090),('default',1024,2048)]:
 r=json.loads((B/f'pilot_{name}.json').read_text());assert r['passed'] and not r.get('error')
 assert r['only_charged_prefill_cleanup_changed'] and r['config']['decode']==5 and r['config']['blocks']==0
 assert r['config']['hot']==hot and r['config']['cold']==cold and len(r['rows'])==2
 assert all(manifest[p]==v for p,v in r['source_sha256'].items())
 a,b=r['rows'];assert a['arm']=='control' and b['arm']=='cleanup'
 for field in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls'):
  assert a[field]==b[field],(name,field)
 for field in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):assert a['trials'][0][field]==b['trials'][0][field]
 assert len(a['every_visible_state_sha256'])==6
 mem=[row['whole_decode_resources'] for row in (a,b)]
 for item in mem:
  assert item['root']['requested_storage_bytes']==item['root']['allocator_block_bytes']==(cold+64)*131072
 reduction=100*(1-mem[1]['gpu_reserved_peak_bytes']/mem[0]['gpu_reserved_peak_bytes'])
 ttft=100*(b['ttft_ms']/a['ttft_ms']-1)
 median=100*(b['itl_ms']/a['itl_ms']-1)
 p95=100*(np.percentile(b['trials'][0]['step_ms'],95)/np.percentile(a['trials'][0]['step_ms'],95)-1)
 assert reduction>=10 and ttft<=10 and median<=2 and p95<=2
 assert r['exact_allocation_stats']['live_allocations']==r['exact_allocation_stats']['live_requested_bytes']==0
 results[name]=dict(gpu_reserved_peak_bytes=[item['gpu_reserved_peak_bytes'] for item in mem],reserved_reduction_pct=reduction,
   ttft_change_pct=ttft,median_decode_change_pct=median,p95_decode_change_pct=p95,
   candidate_active_plus_host_mib=mem[1]['allocated_extra_plus_host_range_bytes']/2**20,
   candidate_reserved_plus_host_mib=mem[1]['reserved_extra_plus_host_range_bytes']/2**20)
report=dict(verified=True,geometries=results,advance_to_three_abba=True,candidate_ready_for_cli=False,
 limitations='Short state-hash-instrumented ordered pair; cold/warm order and variance preclude timing claim. Complete resource target still fails. No active/model-wide memory, speed, quality, serving or physical-STT claim.',
 evidence_sha256={p:sha(B/p) for p in ['plan.json','candidate_source.tar.gz','candidate_manifest.json','pilot_shared.json','pilot_default.json']},verifier_sha256=sha(Path(__file__)))
out=B/'verified_pilot.json';assert not out.exists();out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
