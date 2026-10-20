"""Recompute default-geometry integrity, request ownership and timing guards."""
import hashlib,json,statistics,tarfile
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
read=lambda name:json.loads((P/name).read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=read('source_manifest.json');native=read('native_manifest.json')
for manifest,bundle in ((source,'source.tar.gz'),(native,'native_source.tar.gz')):
 assert sha(P/bundle)==manifest['bundle_sha256']
 with tarfile.open(P/bundle) as archive:
  for name,expected in manifest['source_sha256'].items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected
assert source['source_sha256']['experiments/bounded_pinned_pool.py']==native['source_sha256']['experiments/bounded_pinned_pool.py']
a,b=read('memory_control.json'),read('memory_candidate.json')
for raw,expected in ((a,402784256),(b,276824064)):
 assert raw['passed'] and raw['hot']==1024 and raw['cold']==2048
 for name,expected_sha in raw['source_sha256'].items():assert expected_sha==source['source_sha256'][name]
 assert raw['owned_pinned_peak_bytes']==raw['owned_pinned_current_bytes']==raw['reserved_host_bytes']==expected
 assert raw['owned_pinned_after_cleanup_bytes']==0
 assert len([e for e in raw['pinned_root_events'] if e['kind']=='allocate'])==1
for key in ('state_sha256','predicted_token_ids','counters','occupied_host_bytes','input_ids_sha256','teacher_token_ids'):
 assert a[key]==b[key],key

def identity(raw):
 assert raw['passed'] and raw['runner_sha256']==source['source_sha256']['experiments/compare_cpu_batched.py']
 assert raw['bounded_backend_sha256']==source['source_sha256']['experiments/bounded_pinned_pool.py']
 assert raw['config']['hot']==1024 and raw['config']['cold']==2048

def exact(a,b,frames):
 for key in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','native_d2h_calls','native_h2d_calls'):
  assert a[key]==b[key],key
 assert len(a['every_visible_state_sha256'])==frames
 for key in ('predicted_token_ids','paid_writes','writes_saved','promoted','demoted','decode_steps'):
  assert a['trials'][0][key]==b['trials'][0][key],key
 assert b['reserved_host_bytes']==(402784256 if b['arm']=='direct_scores' else 276824064)
parity=read('parity.json');identity(parity);assert len(parity['rows'])==8
for index,ctx in enumerate((4096,8192,16384,31500)):
 x,y=parity['rows'][index*2:index*2+2];assert x['context']==y['context']==ctx;exact(x,y,6)
raw=read('abba.json');identity(raw);assert len(raw['rows'])==12
assert [r['arm'] for r in raw['rows']]==['direct_scores','bounded_scores','bounded_scores','direct_scores']*3
assert raw['config']['warmup']==1 and raw['config']['decode']==50
for index,row in enumerate(raw['rows']):
 assert row['block']==index//4 and row['context']==16384;exact(raw['rows'][0],row,0)
summary={}
for arm in ('direct_scores','bounded_scores'):
 rows=[r for r in raw['rows'] if r['arm']==arm];steps=[v for r in rows for v in r['trials'][0]['step_ms']];assert len(steps)==300
 summary[arm]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)),ttft_ms=statistics.median(r['ttft_ms'] for r in rows),prefill_peak_gb=max(r['prefill_peak_memory_gb'] for r in rows))
x,y=summary['direct_scores'],summary['bounded_scores']
s=read('native.json');assert s['passed'] and s['default_geometry'] and len(s['checks'])==4
for name,expected in s['source_sha256'].items():assert expected==native['source_sha256'][name]
assert max(r['transient_logical_peak'] for r in s['checks'])==2112
assert all(r['exact_state_payload_counters'] and r['rejected_append_before_mutation'] and r['appended_tokens']==210 for r in s['checks'])
gates=dict(p95_regression_at_most_2pct=y['p95_ms']<=1.02*x['p95_ms'],ttft_regression_at_most_10pct=y['ttft_ms']<=1.1*x['ttft_ms'],prefill_peak_regression_at_most_2pct=y['prefill_peak_gb']<=1.02*x['prefill_peak_gb'])
out=dict(source_integrity_memory_and_native_bounds_verified=True,summary=summary,regression_gates=gates,regression_gates_passed=all(gates.values()),owned_root_request_reduction_pct=100*(1-b['owned_pinned_peak_bytes']/a['owned_pinned_peak_bytes']),sampled_rss_peak_control_bytes=max(r['VmRSS_bytes'] for r in a['rss_samples']),sampled_rss_peak_candidate_bytes=max(r['VmRSS_bytes'] for r in b['rss_samples']),gpu_peak_control_bytes=a['measurement_final']['gpu_peak_allocated_bytes'],gpu_peak_candidate_bytes=b['measurement_final']['gpu_peak_allocated_bytes'],source_inferred_active_pinned_capacity_bytes_both=536870912,host_allocator_reservation_telemetry=False,default_geometry_freegeneration_interleaved_not_verified=True,limitations='Smaller explicitroot request atusualbudgets; bothallocatorblocks inferred512MiB, sampledRSSdifference notlarge. No newphysicalmemory/speed/quality claim; special991/1090benefit doesnotgeneralize. Fixedteacher/nativechecks only atthese budgets; opt-in/defaults unchanged.',evidence_sha256={name:sha(P/name) for name in ('parity.json','memory_control.json','memory_candidate.json','abba.json','native.json')})
dest=P/'verified_report.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
