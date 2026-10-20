"""Validate raw paired trial against archived source and resource trajectories."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

P=Path(__file__).resolve().parent
read=lambda name:json.loads((P/name).read_text())
raw,plan,audit=read('timing.json'),read('timing_plan.json'),read('result.json')
manifest=read('timing_source_manifest.json')
assert raw['passed'] and audit['passed'] and len(raw['rows'])==12
assert hashlib.sha256((P/'verified_report.json').read_bytes()).hexdigest()==plan['resource_audit_sha256']
assert hashlib.sha256((P/'timing_source.tar.gz').read_bytes()).hexdigest()==manifest['bundle_sha256']
with tarfile.open(P/'timing_source.tar.gz') as archive:
 for name,expected in raw['source_sha256'].items():
  assert expected==manifest['source_sha256'][name]==hashlib.sha256(archive.extractfile(name).read()).hexdigest()
 for name,expected in audit['source_sha256'].items():
  assert expected==manifest['source_sha256'][name],name
for key in ('seed','context','decode','warmup','repeat'):
 assert raw[key]==plan[key]
for key in ('teacher_token_ids','input_token_ids','prompt_sha256','teacher_source_sha256','settings'):
 assert raw[key]==audit[key],key
assert [r['arm'] for r in raw['rows']]==plan['order']
controls={r['arm']:r for r in audit['rows']}
for index,row in enumerate(raw['rows']):
 assert row['block']==index//4 and len(row['step_ms'])==len(row['forward_ms'])==50
 reference=controls[row['arm']]
 assert row['predicted_token_ids']==reference['predicted_token_ids']
 assert row['counters']==reference['counters']
 for key in ('h2d_payload_bytes','d2h_payload_bytes'):
  assert row[key]==reference[key]
 for key in ('layers','distinct_owned_kv_storage_bytes','visible_kv_tensor_payload_bytes'):
  assert row['final_boundary'][key]==reference['boundaries'][-1][key]
 assert sum(row['final_boundary']['distinct_owned_kv_storage_bytes'].values())<=raw['owned_kv_ceiling_bytes']
summary={}
for arm in ('snap','cpu_direct'):
 rows=[r for r in raw['rows'] if r['arm']==arm]
 steps=[v for r in rows for v in r['step_ms']];assert len(steps)==300
 summary[arm]=dict(median_decode_ms=statistics.median(steps),p95_decode_ms=float(np.percentile(steps,95)),
  median_ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
  maximum_prefill_peak_gpu_bytes=max(r['prefill_peak_gpu_bytes'] for r in rows),
  maximum_decode_peak_gpu_bytes=max(r['decode_peak_gpu_bytes'] for r in rows),
  final_owned_kv_bytes=rows[0]['final_boundary']['distinct_owned_kv_storage_bytes'],
  block_median_ms=[statistics.median(v for r in rows if r['block']==block for v in r['step_ms']) for block in range(3)])
a,b=summary['snap'],summary['cpu_direct']
gates=dict(median_gain_5pct=b['median_decode_ms']<=.95*a['median_decode_ms'],
 p95_regression_at_most_2pct=b['p95_decode_ms']<=1.02*a['p95_decode_ms'],
 all_blocks_faster=all(x<y for x,y in zip(b['block_median_ms'],a['block_median_ms'])),
 ttft_regression_at_most_10pct=b['median_ttft_ms']<=1.1*a['median_ttft_ms'])
out=dict(executed_source_and_resource_trajectory_and_repeatability_verified=True,
 summary=summary,performance_gates=gates,all_performance_gates_passed=all(gates.values()),
 cpu_over_snap_decode_ratio=b['median_decode_ms']/a['median_decode_ms'],
 cpu_vs_snap_ttft_change_pct=100*(b['median_ttft_ms']/a['median_ttft_ms']-1),
 shared_resource_ceiling=True,actual_usage_equal=False,joint_quality_gate=False,
 timing_sha256=hashlib.sha256((P/'timing.json').read_bytes()).hexdigest(),
 source_bundle_sha256=manifest['bundle_sha256'],limitations=raw['limitations'])
dest=P/'verified_timing.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
