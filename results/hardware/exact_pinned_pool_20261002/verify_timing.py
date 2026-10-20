"""Recompute three-ABBA regression gates for exact versus cached pinned roots."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import numpy as np
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    primary=json.loads((BASE/'verified_memory.json').read_text());assert primary['verified']
    hashes={}
    with tarfile.open(BASE/'source_model_v2.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile():hashes[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    results={};evidence={}
    for geometry,hot,cold,request in (('shared',991,1090,151257088),('default',1024,2048,276824064)):
        path=BASE/f'abba_{geometry}.json';raw=json.loads(path.read_text());evidence[path.name]=sha(path)
        assert raw['passed'] and raw['exact_root_comparison'] and len(raw['rows'])==12
        assert all(hashes[p]==value for p,value in raw['exact_source_sha256'].items())
        cfg=raw['config'];assert cfg['contexts']==[16384] and cfg['hot']==hot and cfg['cold']==cold
        assert cfg['blocks']==3 and cfg['decode']==50 and cfg['warmup']==1
        assert [r['arm'] for r in raw['rows']]==['fused_bounded','native_reuse','native_reuse','fused_bounded']*3
        reference=raw['rows'][0]
        for index,row in enumerate(raw['rows']):
            assert row['block']==index//4 and row['context']==16384 and row['reserved_host_bytes']==request
            for field in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes',
                'explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes',
                'native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls'):
                assert row[field]==reference[field],(geometry,field)
            for field in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):
                assert row['trials'][0][field]==reference['trials'][0][field],(geometry,field)
            if row['arm']=='native_reuse':assert row['reuse_counts']==dict(reused_layers=1600,fallback_layers=0,reused_compactions=1600)
        assert raw['exact_allocation_stats']['live_allocations']==0
        live={}
        for event in raw['exact_allocation_stats']['events']:
            if event['kind']=='allocate':assert event['bytes']==request;live[event['token']]=event['bytes']
            else:assert live.pop(event['token'])==event['bytes']
        assert not live
        arms={}
        for source,label in [('fused_bounded','cached_native_reuse'),('native_reuse','exact_native_reuse')]:
            chosen=[r for r in raw['rows'] if r['arm']==source]
            steps=[v for r in chosen for v in r['trials'][0]['step_ms']];assert len(steps)==300
            arms[label]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)),
                ttft_ms=statistics.median(r['trials'][0]['ttft_ms'] for r in chosen),
                prefill_gpu_peak_gb=max(r['prefill_peak_memory_gb'] for r in chosen),decode_gpu_peak_gb=max(r['peak_memory_gb'] for r in chosen),
                block_median_ms=[statistics.median(v for r in chosen if r['block']==b for v in r['trials'][0]['step_ms']) for b in range(3)])
        a,b=arms['cached_native_reuse'],arms['exact_native_reuse']
        guards=dict(median=b['median_ms']<=1.02*a['median_ms'],p95=b['p95_ms']<=1.02*a['p95_ms'],
            ttft=b['ttft_ms']<=1.10*a['ttft_ms'],prefill_peak=b['prefill_gpu_peak_gb']<=1.02*a['prefill_gpu_peak_gb'])
        results[geometry]=dict(arms=arms,regression_guards=guards,median_change_pct=100*(b['median_ms']/a['median_ms']-1),
            ttft_change_pct=100*(b['ttft_ms']/a['ttft_ms']-1),advance_to_integrity=all(guards.values()))
    result=dict(verified=True,geometries=results,advance_to_mandatory_integrity=all(r['advance_to_integrity'] for r in results.values()),
        no_new_speed_claim=True,memory_primary_sha256=sha(BASE/'verified_memory.json'),evidence_sha256=evidence,
        verifier_sha256=sha(Path(__file__)),limitations='Memory architecture regression check only; same teacher/state/legacy history and synchronization instrumentation. Exact allocation/free work changes request lifecycle; model serving throughput/concurrency/independent quality/full-active-resource/CLI gates remain. No model-wide10%memory/physical STT or new SnapKV superiority claim.')
    path=BASE/'verified_timing.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
