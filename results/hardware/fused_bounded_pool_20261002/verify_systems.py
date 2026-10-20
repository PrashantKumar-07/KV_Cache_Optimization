"""Verify composition source identity, saturation, model parity and guards."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(a,b,frames,old,new):
    for key in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes',
        'explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','native_d2h_calls',
        'native_h2d_calls','per_layer_initial_d2h_calls'):
        assert a[key]==b[key],key
    assert len(a['every_visible_state_sha256'])==len(b['every_visible_state_sha256'])==frames
    assert a['reserved_host_bytes']==old and b['reserved_host_bytes']==new
    for key in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):
        assert a['trials'][0][key]==b['trials'][0][key],key


def main():
    hashes=json.loads((BASE/'source_manifest.json').read_text())['source_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for name,digest in hashes.items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
    prior=json.loads((ROOT/'results/hardware/fused_cli_integration_20261002/source_manifest.json').read_text())['source_sha256']
    for name in ('src/tiered_kv_cache.py','experiments/bounded_pinned_pool.py','experiments/fused_eviction_policy.py',
        'experiments/fused_hot_compaction.py','experiments/native_hot_compact.cpp','experiments/native_hot_compact.cu'):
        assert hashes[name]==prior[name],name
    primary=json.loads((ROOT/'results/hardware/fused_eviction_model_20261002/verified_report.json').read_text())
    cases=[]
    for name,count,peak in (('synthetic',36,1154),('synthetic_default',4,2112)):
        x=json.loads((BASE/f'{name}.json').read_text())
        assert x['passed'] and x['fused'] and len(x['checks'])==count and x['binary_sha256']==primary['binary_sha256']
        for path,digest in x['source_sha256'].items():assert hashes[path]==digest
        assert max(r['transient_logical_peak'] for r in x['checks'])==peak
        assert all(r['exact_state_payload_counters'] and r['rejected_append_before_mutation'] and
            r['last_query_summary'] and r['decode_calls']==24 for r in x['checks'])
        for r in x['checks']:
            assert r['root_request_bytes']==[(r['cold']+r['hot']+1)*8192,r['slot_capacity']*8192]
        cases.extend(x['checks'])
    assert sum(r['fused_compactions'] for r in cases)>0 and sum(r['eager_compactions'] for r in cases)>0
    geometries={}
    for name,hot,cold,old,new in (('default',1024,2048,402784256,276824064),('shared',991,1090,272891904,151257088)):
        parity=json.loads((BASE/f'parity_{name}.json').read_text())
        abba=json.loads((BASE/f'abba_{name}.json').read_text())
        for x in (parity,abba):
            assert x['passed'] and x['config']['hot']==hot and x['config']['cold']==cold and x['config']['real_query_prefill']
            assert x['runner_sha256']==hashes['experiments/compare_cpu_batched.py']
            assert x['fused_binary_sha256']==primary['binary_sha256']
            for path,digest in x['fused_source_sha256'].items():assert hashes[path]==digest
        assert len(parity['rows'])==8
        for i,ctx in enumerate((4096,8192,16384,31500)):
            a,b=parity['rows'][2*i:2*i+2]
            assert a['context']==b['context']==ctx and b['fused_compactions']>0
            exact(a,b,13,old,new)
        assert len(abba['rows'])==12 and abba['config']['blocks']==3 and abba['config']['warmup']==1 and abba['config']['decode']==50
        rows=abba['rows'];control,candidate='fused_scores','fused_bounded_scores'
        for block in range(3):
            chosen=rows[4*block:4*block+4]
            assert [r['arm'] for r in chosen]==[control,candidate,candidate,control]
        reference=rows[0]
        for row in rows:
            exact(reference,row,0,old,new if row['arm']==candidate else old)
        summary={}
        for arm in (control,candidate):
            chosen=[r for r in rows if r['arm']==arm]
            steps=[v for r in chosen for v in r['trials'][0]['step_ms']]
            assert len(steps)==300
            summary[arm]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)),
                ttft_ms=statistics.median(r['trials'][0]['ttft_ms'] for r in chosen),
                prefill_peak_gb=max(r['prefill_peak_memory_gb'] for r in chosen))
        a,b=summary[control],summary[candidate]
        guards=dict(median=b['median_ms']<=1.02*a['median_ms'],p95=b['p95_ms']<=1.02*a['p95_ms'],
            ttft=b['ttft_ms']<=1.10*a['ttft_ms'],prefill_peak=b['prefill_peak_gb']<=1.02*a['prefill_peak_gb'])
        geometries[name]=dict(hot=hot,cold=cold,root_request_bytes=dict(fused=old,combined=new),
            request_reduction_pct=100*(1-new/old),summary=summary,regression_guards=guards,
            advance_to_stress=all(guards.values()))
    result=dict(verified=True,components_shader_and_math_unchanged=True,native_cases=40,
        full_transient_peaks=[1154,2112],fused_compactions=sum(r['fused_compactions'] for r in cases),
        eager_compactions=sum(r['eager_compactions'] for r in cases),geometries=geometries,
        advance_to_stress=all(r['advance_to_stress'] for r in geometries.values()),
        evidence_sha256={p.name:sha(p) for p in BASE.glob('*.json') if p.name!='verified_systems.json'},
        source_archive_sha256=sha(BASE/'source.tar.gz'),verifier_sha256=sha(Path(__file__)),
        limitations='Owned pinned request reduction, samelogical policy/quality. Requestbytes do not directly measure allocatorcapacity/RSS/physicalSTT; defaultrootrequests still roundto512MiB. Speedguard preservation only, not new5%speedclaim; mandatoryfreegeneration/isolation/CLI before ordinary combination.')
    output=BASE/'verified_systems.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
