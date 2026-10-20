"""Independently verify fixed-stream parity, source identity and complete-loop gate."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
root=Path('results/hardware/realq_cpu_systems_20261001');output=root/'verified_gate_report.json'
assert not output.exists(),'Preserve prior evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads((root/'source_manifest.json').read_text())
assert sha(root/'source.tar.gz')==source['bundle_sha256']
for path,expected in source['source_sha256'].items():assert sha(path)==expected,path
parity=json.loads((root/'parity.json').read_text());raw=json.loads((root/'abba_16k_50.json').read_text())
assert parity['passed'] and raw['passed']
for r in (parity,raw):
    assert r['config']['control']=='gpu' and r['config']['candidate']=='batched_scores' and r['config']['real_query_prefill']
    assert r['exact_model_policy_state'] and r['runner_sha256']==source['source_sha256']['experiments/compare_cpu_batched.py']
    assert r['query_capture_sha256']==source['source_sha256']['experiments/evaluate_query_selection.py']
    for field,path in [('backend_sha256','cpu_batched_cache.py'),('slot_backend_sha256','cpu_slot_cache.py'),('score_backend_sha256','batched_score_policy.py')]:
        assert r[field]==source['source_sha256'][f'experiments/{path}']
assert parity['config']['contexts']==[4096,8192,16384] and parity['config']['decode']==5 and parity['config']['blocks']==0
assert len(parity['rows'])==6
fields=('state_sha256','history_sha256')
for context in (4096,8192,16384):
    pair={row['arm']:row for row in parity['rows'] if row['context']==context}
    assert set(pair)=={'gpu','batched_scores'}
    for field in fields+('every_visible_state_sha256',):assert pair['gpu'][field]==pair['batched_scores'][field]
assert raw['config']['contexts']==[16384] and raw['config']['decode']==50 and raw['config']['blocks']==3 and raw['config']['warmup']==1
assert len(raw['rows'])==12
expected=[(block,arm) for block in range(3) for arm in ('gpu','batched_scores','batched_scores','gpu')]
assert [(row['block'],row['arm']) for row in raw['rows']]==expected
reference=raw['rows'][0]
for row in raw['rows']:
    for field in fields:assert row[field]==reference[field]
    assert len(row['trials'])==1
    for a,b in zip(row['trials'],reference['trials']):
        assert len(a['step_ms'])==50
        for field in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):assert a[field]==b[field]
latency={};resources={}
for arm in ('gpu','batched_scores'):
    rows=[row for row in raw['rows'] if row['arm']==arm]
    steps=[ms for row in rows for trial in row['trials'] for ms in trial['step_ms']]
    latency[arm]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)))
    resources[arm]=dict(median_ttft_ms=statistics.median(row['ttft_ms'] for row in rows),
        max_prefill_peak_gpu_bytes=round(max(row['prefill_peak_memory_gb'] for row in rows)*1e9),
        max_decode_peak_gpu_bytes=round(max(row['peak_memory_gb'] for row in rows)*1e9),
        median_final_gpu_allocated_bytes=statistics.median(row['final_gpu_allocated_bytes'] for row in rows),
        occupied_host_bytes=sorted({row['occupied_host_bytes'] for row in rows}),reserved_host_bytes=sorted({row['reserved_host_bytes'] for row in rows}),
        h2d_bytes=sorted({row['explicit_h2d_payload_bytes'] for row in rows}),d2h_bytes=sorted({row['explicit_d2h_payload_bytes'] for row in rows}))
blocks=[{arm:statistics.median(ms for row in raw['rows'] if row['block']==block and row['arm']==arm for trial in row['trials'] for ms in trial['step_ms']) for arm in ('gpu','batched_scores')} for block in range(3)]
gate=latency['batched_scores']['median_ms']<=.95*latency['gpu']['median_ms'] and latency['batched_scores']['p95_ms']<=1.02*latency['gpu']['p95_ms'] and all(block['batched_scores']<block['gpu'] for block in blocks)
assert raw['summary']['16384']['latency_gate_passed']==gate
report=dict(source_and_parity_and_state_verified=True,raw_sha256=sha(root/'abba_16k_50.json'),parity_sha256=sha(root/'parity.json'),source_bundle_sha256=source['bundle_sha256'],
    latency=latency,blocks=blocks,latency_gate_passed=gate,
    gain_percent=100*(1-latency['batched_scores']['median_ms']/latency['gpu']['median_ms']),resources=resources,
    ttft_change_percent=100*(resources['batched_scores']['median_ttft_ms']/resources['gpu']['median_ttft_ms']-1),
    ttft_guard_passed=resources['batched_scores']['median_ttft_ms']<=1.1*resources['gpu']['median_ttft_ms'],
    limitations='One fixed16K50teacher stream/three paired blocks; conditional same-placement backend evidence. Physical host resources unequal. Not faithful SnapKV speed, serving throughput, model-wide peak reduction, controller/STT or independent headline replication.')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
