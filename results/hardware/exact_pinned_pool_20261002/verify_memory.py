"""Verify exact-pool model state, root ranges and fresh-process memory."""
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(name):
    values={}
    with tarfile.open(BASE/name) as archive:
        for member in archive.getmembers():
            if member.isfile():values[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return values
def balanced(events):
    live={}
    for event in events:
        if event['kind']=='allocate':assert event['token'] not in live;live[event['token']]=event['bytes']
        else:assert live.pop(event['token'])==event['bytes']
    assert not live

def main():
    native=json.loads((BASE/'verified_native.json').read_text());assert native['verified']
    model_plan=json.loads((BASE/'model_plan.json').read_text())
    assert model_plan['native_proof_sha256']==sha(BASE/'verified_native.json')
    old=manifest('source.tar.gz')|manifest('model_driver_source.tar.gz')
    current=manifest('source_model_v2.tar.gz')
    memory_plan=json.loads((BASE/'memory_plan.json').read_text())
    assert sha(BASE/'source_model_v2.tar.gz')==memory_plan['source_bundle_sha256']
    assert current==json.loads((BASE/'source_model_v2_manifest.json').read_text())
    assert old['experiments/compare_exact_pool.py']==model_plan['driver_sha256']
    report={};evidence={}
    for geometry,hot,cold,request,cached_range in (('shared',991,1090,151257088,268435456),('default',1024,2048,276824064,536870912)):
        parity_path=BASE/f'parity_{geometry}.json';parity=json.loads(parity_path.read_text())
        assert parity['passed'] and parity['exact_root_comparison'] and len(parity['rows'])==8
        assert all(old[p]==value for p,value in parity['exact_source_sha256'].items())
        for context in (4096,8192,16384,31500):
            a,b=[r for r in parity['rows'] if r['context']==context]
            assert a['arm']=='fused_bounded' and b['arm']=='native_reuse'
            for field in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes',
                'explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes','reserved_host_bytes',
                'native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls'):
                assert a[field]==b[field],(geometry,context,field)
            assert len(a['every_visible_state_sha256'])==13 and a['reserved_host_bytes']==request
            for field in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):
                assert a['trials'][0][field]==b['trials'][0][field],(geometry,context,field)
            assert b['reuse_counts']==dict(reused_layers=384,fallback_layers=0,reused_compactions=384)
        assert parity['exact_allocation_stats']['live_allocations']==0
        balanced(parity['exact_allocation_stats']['events'])
        rows={}
        for arm in ('cached','exact'):
            path=BASE/f'memory_{geometry}_{arm}.json';raw=json.loads(path.read_text())
            assert raw['passed'] and raw['actual_arm']=='native_reuse' and raw['hot']==hot and raw['cold']==cold
            assert all(current[p]==value for p,value in raw['actual_source_sha256'].items())
            assert len(raw['boundaries'])==52 and len(raw['predicted_token_ids'])==50
            assert raw['owned_pinned_peak_bytes']==raw['reserved_host_bytes']==request
            capacity=cached_range if arm=='cached' else request
            assert raw['owned_host_allocator_block_peak_bytes']==capacity
            assert raw['host_block_observations'][0]['allocator_block_bytes']==capacity
            assert raw['owned_pinned_after_cleanup_bytes']==raw['owned_host_allocator_block_after_cleanup_bytes']==0
            if arm=='exact':
                assert raw['exact_root_memory'] and raw['exact_allocation_stats']['live_allocations']==0
                balanced(raw['exact_allocation_stats']['events'])
            rows[arm]=raw;evidence[path.name]=sha(path)
        a,b=rows['cached'],rows['exact']
        for field in ('state_sha256','predicted_token_ids','counters','occupied_host_bytes','reserved_host_bytes','prompt_sha256',
            'teacher_source_sha256','teacher_token_ids','input_ids_sha256','gpu_phase_peaks'):
            assert a[field]==b[field],(geometry,field)
        rss={arm:max(row['VmRSS_bytes'] for row in raw['rss_samples']) for arm,raw in rows.items()}
        report[geometry]=dict(exact_four_context_parity=True,boundary_frames_per_arm=52,
            root_request_bytes=request,cached_root_range_bytes=cached_range,exact_root_range_bytes=request,
            host_range_reduction_pct=100*(1-request/cached_range),
            sampled_peak_rss_bytes=rss,sampled_peak_rss_reduction_bytes=rss['cached']-rss['exact'],
            whole_gpu_peak_bytes=b['gpu_phase_peaks']['whole_allocated_peak'],whole_gpu_peak_reduction_pct=0,
            owned_roots_after_cleanup=0)
        evidence[parity_path.name]=sha(parity_path)
    assert '198 passed' in (BASE/'full_suite.txt').read_text()
    result=dict(verified=True,geometries=report,full_suite_tests=198,fresh_memory_processes=4,
        advance_to_model_regression_and_integrity=True,complete_actual_resource_equality=False,
        native_sha256=sha(BASE/'verified_native.json'),source_v2_bundle_sha256=sha(BASE/'source_model_v2.tar.gz'),
        evidence_sha256=evidence,suite_sha256=sha(BASE/'full_suite.txt'),verifier_sha256=sha(Path(__file__)),
        limitations='Same capacities, exact state/IDs/counters/native payloads/calls. Host CUDA allocation ranges shrink44%/48%; sampled fresh RSS descriptive oneprocessperarm/geometry, not CI. GPU peak unchanged. Known policy-only corrected root padding is removed, but complete forward/OS/cached memory budget not verified; no actual equality/model-wide10%memory/speed/quality/physical-STT claim. ABBA/generation/recycling/active-resource/CLI gates remain.')
    path=BASE/'verified_memory.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
