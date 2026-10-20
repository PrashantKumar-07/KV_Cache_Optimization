"""Verify bounded-pool source, exact state, ownership and regression guards."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

P=Path(__file__).resolve().parent
read=lambda name:json.loads((P/name).read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle(prefix):
 m=read(prefix+'_manifest.json')
 assert sha(P/(prefix+'_source.tar.gz'))==m['bundle_sha256']
 with tarfile.open(P/(prefix+'_source.tar.gz')) as archive:
  for name,expected in m['source_sha256'].items():
   assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
 return m


synthetic_source=bundle('synthetic');model_source=bundle('model')
assert synthetic_source['source_sha256']['experiments/bounded_pinned_pool.py']==model_source['source_sha256']['experiments/bounded_pinned_pool.py']
synthetic=read('synthetic.json');assert synthetic['passed'] and len(synthetic['checks'])==32
for name,expected in synthetic['source_sha256'].items():assert expected==synthetic_source['source_sha256'][name]
assert synthetic['full_transient_saturation_observed']
assert max(row['transient_logical_peak'] for row in synthetic['checks'])==75
assert all(row['appended_tokens']==36 and row['exact_state_payload_counters'] and row['rejected_append_before_mutation'] for row in synthetic['checks'])


def identity(report):
 assert report['runner_sha256']==model_source['source_sha256']['experiments/compare_cpu_batched.py']
 for key,name in (('bounded_backend_sha256','bounded_pinned_pool'),('direct_backend_sha256','direct_shared_pool'),
                  ('backend_sha256','cpu_batched_cache'),('slot_backend_sha256','cpu_slot_cache'),
                  ('query_capture_sha256','evaluate_query_selection'),('score_backend_sha256','batched_score_policy')):
  assert report[key]==model_source['source_sha256'][f'experiments/{name}.py']
 assert report['config']['hot']==991 and report['config']['cold']==1090


def exact(a,b,frames):
 for key in ('state_sha256','history_sha256','every_visible_state_sha256',
             'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes',
             'occupied_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls'):
  assert a[key]==b[key],key
 assert len(a['every_visible_state_sha256'])==frames
 assert len(a['trials'])==len(b['trials'])==1
 for key in ('predicted_token_ids','paid_writes','writes_saved','promoted','demoted','decode_steps'):
  assert a['trials'][0][key]==b['trials'][0][key],key


parity=read('parity.json');identity(parity)
assert parity['passed'] and len(parity['rows'])==8
for index,context in enumerate((4096,8192,16384,31500)):
 a,b=parity['rows'][2*index:2*index+2]
 assert a['context']==b['context']==context
 assert a['reserved_host_bytes']==272891904 and b['reserved_host_bytes']==151257088
 exact(a,b,6)
control,candidate=read('memory_control.json'),read('memory_candidate.json')
for report,expected in ((control,272891904),(candidate,151257088)):
 assert report['passed'] and report['hot']==991 and report['cold']==1090
 for name,value in report['source_sha256'].items():assert value==model_source['source_sha256'][name]
 assert report['owned_pinned_peak_bytes']==report['owned_pinned_current_bytes']==report['reserved_host_bytes']==expected
 allocations=[event for event in report['pinned_root_events'] if event['kind']=='allocate']
 assert len(allocations)==1 and allocations[0]['bytes']==expected
 assert report['owned_pinned_after_cleanup_bytes']==0
 assert len(report['predicted_token_ids'])==50 and len(report['boundaries'])==52
for key in ('state_sha256','predicted_token_ids','counters','occupied_host_bytes',
            'input_ids_sha256','teacher_token_ids','prompt_sha256','teacher_source_sha256'):
 assert control[key]==candidate[key],key
reduction=100*(1-candidate['owned_pinned_peak_bytes']/control['owned_pinned_peak_bytes'])
abba=read('abba.json');identity(abba)
assert abba['passed'] and len(abba['rows'])==12 and abba['config']['warmup']==1 and abba['config']['decode']==50
assert [row['arm'] for row in abba['rows']]==['direct_scores','bounded_scores','bounded_scores','direct_scores']*3
for index,row in enumerate(abba['rows']):
 assert row['context']==16384 and row['block']==index//4
 exact(abba['rows'][0],row,0)
 assert row['reserved_host_bytes']==(272891904 if row['arm']=='direct_scores' else 151257088)
summary={}
for arm in ('direct_scores','bounded_scores'):
 rows=[row for row in abba['rows'] if row['arm']==arm]
 steps=[v for row in rows for v in row['trials'][0]['step_ms']];assert len(steps)==300
 summary[arm]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)),
  ttft_ms=statistics.median(row['ttft_ms'] for row in rows),prefill_peak_gb=max(row['prefill_peak_memory_gb'] for row in rows))
a,b=summary['direct_scores'],summary['bounded_scores']
gates=dict(owned_pinned_reduction_at_least_25pct=reduction>=25,
 p95_regression_at_most_2pct=b['p95_ms']<=1.02*a['p95_ms'],
 ttft_regression_at_most_10pct=b['ttft_ms']<=1.1*a['ttft_ms'],
 prefill_gpu_peak_regression_at_most_2pct=b['prefill_peak_gb']<=1.02*a['prefill_peak_gb'])
output=dict(source_state_payload_and_ownership_verified=True,gates=gates,primary_and_regression_gates_passed=all(gates.values()),
 owned_pinned_reduction_pct=reduction,summary=summary,
 decode_change_pct=100*(b['median_ms']/a['median_ms']-1),p95_change_pct=100*(b['p95_ms']/a['p95_ms']-1),
 ttft_change_pct=100*(b['ttft_ms']/a['ttft_ms']-1),
 memory=dict(control_request_bytes=control['owned_pinned_peak_bytes'],candidate_request_bytes=candidate['owned_pinned_peak_bytes'],
  sampled_rss_peak_control_bytes=max(row['VmRSS_bytes'] for row in control['rss_samples']),
  sampled_rss_peak_candidate_bytes=max(row['VmRSS_bytes'] for row in candidate['rss_samples']),
  gpu_peak_control_bytes=control['measurement_final']['gpu_peak_allocated_bytes'],
  gpu_peak_candidate_bytes=candidate['measurement_final']['gpu_peak_allocated_bytes'],
  source_inferred_pinned_root_capacity_control_bytes=536870912,source_inferred_pinned_root_capacity_candidate_bytes=268435456,
  host_allocator_reservation_measured=False),
 mandatory_remaining='Free generation and interleaved/recycled model stress if regression gates pass.',
 limitations='One model/seed/fixed teacher stream. Requested host-root memory improvement; unchanged model GPU peak, no10%model memory saving. Allocator capacity inferred, not direct telemetry; cached blocks excluded. Existing991/1090quality−2.57versusSnap and NarrativeNI failures unchanged; no defaultpromotion/serving/physicalSTT/broadcompletion claim.',
 evidence_sha256={name:sha(P/name) for name in ('synthetic.json','parity.json','memory_control.json','memory_candidate.json','abba.json')})
dest=P/'verified_report.json';assert not dest.exists();dest.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
