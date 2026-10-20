import hashlib,json,statistics,tarfile
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
output=p/'verified_report.json';assert not output.exists()
initial=json.loads((p/'source_manifest.json').read_text());timed=json.loads((p/'abba_source_manifest.json').read_text())
for meta,name in ((initial,'source.tar.gz'),(timed,'abba_source.tar.gz')):
 assert sha(p/name)==meta['bundle_sha256']
 with tarfile.open(p/name) as a:
  for path,h in meta['source_sha256'].items():assert hashlib.sha256(a.extractfile(path).read()).hexdigest()==h
parity=json.loads((p/'parity.json').read_text());raw=json.loads((p/'abba.json').read_text())
control,candidate=[json.loads((p/f'memory_{x}.json').read_text()) for x in ('control','candidate')]
synthetic=json.loads((p/'synthetic.json').read_text())
assert all(r['passed'] for r in (parity,raw,control,candidate,synthetic))
assert len(synthetic['checks'])==32 and all(x['exact_state_payload_counters_and_cleanup'] for x in synthetic['checks'])
assert len(parity['rows'])==6
for r,meta in ((parity,initial),(raw,timed)):
 assert r['runner_sha256']==meta['source_sha256']['experiments/compare_cpu_batched.py']
 assert r['direct_backend_sha256']==meta['source_sha256']['experiments/direct_shared_pool.py']
 assert r['query_capture_sha256']==meta['source_sha256']['experiments/evaluate_query_selection.py']
 assert r['config']['control']=='batched_scores' and r['config']['candidate']=='direct_scores' and r['config']['real_query_prefill']
for ctx in (4096,8192,16384):
 pair={x['arm']:x for x in parity['rows'] if x['context']==ctx}
 for k in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_bytes'):
  assert pair['batched_scores'][k]==pair['direct_scores'][k]
 for r in pair.values():assert len(r['every_visible_state_sha256'])==6
for r in (control,candidate):
 assert r['source_sha256']=={k:initial['source_sha256'][k] for k in r['source_sha256']}
 assert r['owned_pinned_after_cleanup_bytes']==0
 current=peak=rounded=rounded_peak=0
 for e in r['pinned_root_events']:
  sign=1 if e['kind']=='allocate' else -1
  current+=sign*e['bytes'];rounded+=sign*(1<<(e['bytes']-1).bit_length())
  assert current==e['owned_bytes'] and current>=0
  peak=max(peak,current);rounded_peak=max(rounded_peak,rounded)
 assert current==rounded==0 and peak==r['owned_pinned_peak_bytes']
 r['trace_inferred_rounded_owned_peak_capacity_bytes']=rounded_peak
for k in ('runtime','source_sha256','input_ids_sha256','teacher_token_ids','predicted_token_ids','state_sha256','counters'):
 assert control[k]==candidate[k],k
assert control['pid']!=candidate['pid']
assert control['owned_pinned_peak_bytes']==805568512 and candidate['owned_pinned_peak_bytes']==402784256
assert control['owned_pinned_current_bytes']==candidate['owned_pinned_current_bytes']==402784256
assert sum(e['kind']=='allocate' for e in candidate['pinned_root_events'])==1
assert len(raw['rows'])==12 and raw['config']['decode']==50 and raw['config']['warmup']==1
assert [(x['block'],x['arm']) for x in raw['rows']]==[(b,a) for b in range(3) for a in ('batched_scores','direct_scores','direct_scores','batched_scores')]
reference=raw['rows'][0]
for r in raw['rows']:
 for k in ('state_sha256','history_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes'):assert r[k]==reference[k]
 assert len(r['trials'])==1 and len(r['trials'][0]['step_ms'])==50
 for k in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):assert r['trials'][0][k]==reference['trials'][0][k]
 if r['arm']=='direct_scores':assert r['per_layer_initial_d2h_calls']==0
metrics={}
for arm in ('batched_scores','direct_scores'):
 rows=[x for x in raw['rows'] if x['arm']==arm]
 steps=[v for x in rows for t in x['trials'] for v in t['step_ms']]
 metrics[arm]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)),median_ttft_ms=statistics.median(x['ttft_ms'] for x in rows),prefill_peak_gpu_bytes=round(max(x['prefill_peak_memory_gb'] for x in rows)*1e9))
a,b=metrics['batched_scores'],metrics['direct_scores']
report=dict(verified=True,raw_sha256={n:sha(p/n) for n in ('parity.json','abba.json','memory_control.json','memory_candidate.json','synthetic.json')},
 exact_parity_and_cleanup=True,memory={name:dict(owned_pinned_peak_bytes=r['owned_pinned_peak_bytes'],trace_inferred_rounded_owned_peak_capacity_bytes=r['trace_inferred_rounded_owned_peak_capacity_bytes'],sampled_generation_rss_peak_bytes=max(x['VmRSS_bytes'] for x in r['rss_samples']),gpu_peak_allocated_bytes=r['measurement_final']['gpu_peak_allocated_bytes']) for name,r in (('control',control),('candidate',candidate))},
 owned_pinned_peak_reduction_percent=100*(1-candidate['owned_pinned_peak_bytes']/control['owned_pinned_peak_bytes']),timing=metrics,
 ttft_change_percent=100*(b['median_ttft_ms']/a['median_ttft_ms']-1),ttft_guard_passed=b['median_ttft_ms']<=1.1*a['median_ttft_ms'],
 decode_median_change_percent=100*(b['median_ms']/a['median_ms']-1),decode_p95_change_percent=100*(b['p95_ms']/a['p95_ms']-1),
 decode_no_more_than_two_percent_regression=b['median_ms']<=1.02*a['median_ms'] and b['p95_ms']<=1.02*a['p95_ms'],
 gpu_peak_guard_passed=b['prefill_peak_gpu_bytes']<=1.02*a['prefill_peak_gpu_bytes'],
 blocks=[{arm:statistics.median(v for x in raw['rows'] if x['block']==block and x['arm']==arm for t in x['trials'] for v in t['step_ms']) for arm in ('batched_scores','direct_scores')} for block in range(3)],
 limitations='Initial ownership/conditional host memory improvement, not overall GPU memory/quality/default promotion. One stream/model/seed; rounded allocator capacity inferred from installed header, not reservation telemetry. RSS sampling lowerbounds. NarrativeQA NI unproven; no faithfulSnap/threadedserving/physicalSTT claim.')
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
