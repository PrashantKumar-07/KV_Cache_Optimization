"""Recompute model regression gates for the exact storage-only candidate."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from verify_parity_resource import archived

BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    primary=json.loads((BASE/'verified_parity_resource.json').read_text())
    plan=json.loads((BASE/'timing_plan.json').read_text())
    assert primary['verified'] and primary['advance_to_whole_model_regression_gates']
    assert sha(BASE/'verified_parity_resource.json')==plan['resource_proof_sha256']
    assert sha(BASE/'full_suite.txt')==plan['full_suite_sha256']
    assert '182 passed' in (BASE/'full_suite.txt').read_text()
    summaries={}
    for name,hot,cold,root in (('shared',991,1090,151257088),('default',1024,2048,276824064)):
        raw=json.loads((BASE/f'abba_{name}.json').read_text())
        assert raw['passed'] and len(raw['rows'])==12
        archived(raw,'_resource')
        assert raw['reuse_binary_sha256']==primary['reuse_binary_sha256']
        cfg=raw['config']
        assert cfg['hot']==hot and cfg['cold']==cold and cfg['contexts']==[16384]
        assert cfg['blocks']==3 and cfg['decode']==50 and cfg['warmup']==1
        assert [r['arm'] for r in raw['rows']]==['fused_bounded','native_reuse','native_reuse','fused_bounded']*3
        reference=raw['rows'][0]
        for index,row in enumerate(raw['rows']):
            assert row['block']==index//4 and row['context']==16384
            assert row['reserved_host_bytes']==root and row['every_visible_state_sha256']==[]
            for field in ('state_sha256','history_sha256','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes',
                'initial_d2h_payload_bytes','occupied_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls'):
                assert reference[field]==row[field],field
            for field in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):
                assert reference['trials'][0][field]==row['trials'][0][field],field
            if row['arm']=='native_reuse':
                assert row['reuse_counts']=={'reused_layers':1600,'fallback_layers':0,'reused_compactions':1600}
        summary={}
        for arm in ('fused_bounded','native_reuse'):
            chosen=[r for r in raw['rows'] if r['arm']==arm]
            values=[v for r in chosen for v in r['trials'][0]['step_ms']];assert len(values)==300
            summary[arm]=dict(median_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),
                ttft_ms=statistics.median(r['trials'][0]['ttft_ms'] for r in chosen),
                prefill_gpu_peak_gb=max(r['prefill_peak_memory_gb'] for r in chosen),
                decode_gpu_peak_gb=max(r['peak_memory_gb'] for r in chosen),
                block_median_ms=[statistics.median(v for r in chosen if r['block']==b for v in r['trials'][0]['step_ms']) for b in range(3)])
        a,b=summary['fused_bounded'],summary['native_reuse']
        guards=dict(median=b['median_ms']<=1.02*a['median_ms'],p95=b['p95_ms']<=1.02*a['p95_ms'],
            ttft=b['ttft_ms']<=1.10*a['ttft_ms'],prefill_peak=b['prefill_gpu_peak_gb']<=1.02*a['prefill_gpu_peak_gb'])
        summaries[name]=dict(summary=summary,regression_guards=guards,advance_to_integrity=all(guards.values()),
            decode_median_change_pct=100*(b['median_ms']/a['median_ms']-1),
            whole_request_peak_change_pct=100*(b['prefill_gpu_peak_gb']/a['prefill_gpu_peak_gb']-1))
    result=dict(verified=True,geometries=summaries,
        advance_to_mandatory_integrity=all(r['advance_to_integrity'] for r in summaries.values()),
        stage_4ms_speed_gate_passed=False,full_suite_tests=182,
        primary_sha256=sha(BASE/'verified_parity_resource.json'),verifier_sha256=sha(Path(__file__)),
        evidence_sha256={name:sha(BASE/name) for name in ('abba_shared.json','abba_default.json','timing_plan.json','full_suite.txt')},
        limitations='Resource candidate regression checks; no new4ms-stage/5%macro speedclaim, independent quality or SnapKV superiority. Prefill unchanged; whole-requestpeak includes originalfullprefill, not just decode/knownKV. Legacy history/synchronization instrumented singlemodel/seed/teacherstream. Native delayedstream/freegeneration/interleaved/CLI gates mandatory beforeordinaryadoption.')
    output=BASE/'verified_timing.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
