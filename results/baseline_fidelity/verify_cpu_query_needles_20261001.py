"""Verify integrated CPU actual-query evidence against frozen GPU output."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path('results/baseline_fidelity')
raw=root/'cpu_query_needles_20261001.json'
ref=root/'tiered_query_needles_v2_20261001.json'
manifest=Path('results/hardware/batched_scores_boundary_retrieval_20261001/manifest.json')
output=root/'cpu_query_needles_verified_20261001.json'
bundle=root/'cpu_query_needles_source_20261001.tar.gz'
assert not output.exists() and not bundle.exists(),'Preserve prior evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads(raw.read_text()); p=json.loads(ref.read_text()); m=json.loads(manifest.read_text())
assert r['passed'] and r['exact_gpu_reference'] and r['reference_sha256']==sha(ref)
assert r['manifest_sha256']==p['manifest_sha256']==sha(manifest)
for path, expected in r['source_sha256'].items():
    assert sha(path)==expected,path
assert sha(root/'check_cpu_query_needles_executed_20261001.py')==r['source_sha256']['experiments/check_cpu_query_needles.py']
controls={row['case']:row for row in p['rows'] if row['arm']=='actual_query'}
cases={case['name']:case for case in m['cases'] if case.get('oracle')}
assert len(r['rows'])==len(controls)==len(cases)==6
rows=[]
for row in r['rows']:
    control=controls[row['case']];case=cases[row['case']]
    assert row['token_sha256']==case['token_sha256']
    for field in ('generated_token_ids','prediction','answer_correct','state_sha256','trajectory_sha256',
                  'trajectory_frames','initial_answer_retention','initial_cache_rows','prefill_query_shapes',
                  'counters','max_cold_rows_per_layer','largest_step_promoted','largest_step_paid_writes'):
        assert row[field]==control[field],field
    assert row['answer_correct']==(case['answer'].casefold() in row['prediction'].casefold())
    assert row['reserved_host_tensor_bytes']==402784256>=row['occupied_host_tensor_bytes']>0
    for field in ('explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes'):
        assert row[field]%4096==0
    rows.append(dict(case=row['case'],answer_correct=row['answer_correct'],
        generated_tokens=len(row['generated_token_ids']),
        complete_span_layers=sum(x['retained_tokens']==12 for x in row['initial_answer_retention'].values()),
        explicit_h2d_payload_bytes=row['explicit_h2d_payload_bytes'],explicit_d2h_payload_bytes=row['explicit_d2h_payload_bytes'],
        occupied_host_tensor_bytes=row['occupied_host_tensor_bytes'],reserved_host_tensor_bytes=row['reserved_host_tensor_bytes']))
assert r['correct_cases']==sum(row['answer_correct'] for row in r['rows'])
files=set(r['source_sha256'])|set(map(str,Path('src').glob('*.py')))|{'experiments/longbench_eval.py','experiments/longbench_metrics.py','experiments/latency_profile.py','experiments/ppl_eval.py','experiments/provenance.py','results/baseline_fidelity/verify_cpu_query_needles_20261001.py'}
digest=hashlib.sha256()
for path in sorted(Path('src').glob('*.py'))+[Path('experiments')/name for name in ('longbench_eval.py','longbench_metrics.py','latency_profile.py','ppl_eval.py','provenance.py')]:
    digest.update(str(path).encode());digest.update(path.read_bytes())
assert digest.hexdigest()==r['runtime']['source_sha256']
with tarfile.open(bundle,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=path)
report=dict(identity_state_output_trajectory_query_and_retention_verified=True,
    raw_sha256=sha(raw),reference_sha256=sha(ref),manifest_sha256=sha(manifest),source_bundle_sha256=sha(bundle),
    exact_gpu_reference=True,correct_cases=r['correct_cases'],rows=rows,
    limitations=r['limitations'])
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
