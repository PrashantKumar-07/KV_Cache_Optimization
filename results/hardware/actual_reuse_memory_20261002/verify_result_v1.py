"""Verify fresh-process root capacities, preservation and corrected KV bound."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((BASE/'plan.json').read_text())
    assert sha(BASE/'source.tar.gz')==plan['source_bundle_sha256']
    assert sha(BASE/'block_preflight.json')==plan['preflight_sha256']
    manifest={}
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile():manifest[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert manifest==json.loads((BASE/'source_manifest.json').read_text())
    pre=json.loads((BASE/'block_preflight.json').read_text())
    assert pre['passed'] and pre['nonpinned_rejection'] and len(pre['checks'])==6
    for row in pre['checks']:
        assert row['allocated'] and row['view_storage_alias_exact']
        assert row['requested_storage_bytes']==row['bytes']
        assert row['allocator_block_bytes']==1<<(row['bytes']-1).bit_length()
    upstream=json.loads((BASE/'upstream_sources.json').read_text())
    for name,entry in upstream.items():assert sha(BASE/name)==entry['sha256']
    header=Path('../venv_kvcache/lib/python3.10/site-packages/torch/include/ATen/core/CachingHostAllocator.h')
    assert header.read_bytes()==(BASE/'CachingHostAllocator.h').read_bytes()
    commands=json.loads((BASE/'commands.json').read_text());assert len(commands)==8 and all(r['returncode']==0 for r in commands)
    summaries={};evidence={}
    fields=('state_sha256','predicted_token_ids','counters','occupied_host_bytes','reserved_host_bytes',
        'prompt_sha256','teacher_source_sha256','teacher_token_ids','input_ids_sha256')
    for geometry,hot,cold,root,block in (('shared',991,1090,151257088,268435456),('default',1024,2048,276824064,536870912)):
        rows={}
        reference=None
        for arm in plan['arms']:
            rows[arm]=[]
            for repeat in range(plan['repeats']):
                name=f'{geometry}_{repeat}_{arm}.json';path=BASE/name;raw=json.loads(path.read_text());evidence[name]=sha(path)
                assert raw['passed'] and raw['actual_arm']==arm and raw['hot']==hot and raw['cold']==cold
                assert raw['host_block_binary_sha256']==pre['binary_sha256']
                assert all(manifest[p]==value for p,value in raw['actual_source_sha256'].items())
                if reference is None:reference=raw
                else:
                    for field in fields:assert raw[field]==reference[field],(name,field)
                assert len(raw['predicted_token_ids'])==50 and len(raw['boundaries'])==52
                assert [r['stage'] for r in raw['boundaries']]==['loaded_model','prefill_compressed']+[f'decode_{i}' for i in range(1,51)]
                assert raw['owned_pinned_peak_bytes']==raw['owned_pinned_current_bytes']==raw['reserved_host_bytes']==root
                assert raw['owned_host_allocator_block_peak_bytes']==block
                assert raw['owned_pinned_after_cleanup_bytes']==raw['owned_host_allocator_block_after_cleanup_bytes']==0
                assert raw['host_block_observations']==[dict(requested_storage_bytes=root,allocator_block_bytes=block,allocated=True,event_count=0,owned_block_capacity_bytes=block)]
                assert all(r['owned_host_allocator_block_bytes']==block for r in raw['boundaries'][1:])
                assert raw['gpu_phase_peaks']['prefill_peak']==raw['gpu_phase_peaks']['whole_allocated_peak']==raw['measurement_final']['gpu_peak_allocated_bytes']
                rows[arm].append(raw)
        arms={}
        for arm,values in rows.items():
            arms[arm]=dict(whole_gpu_allocated_peak_bytes=max(r['gpu_phase_peaks']['whole_allocated_peak'] for r in values),
                whole_gpu_reserved_peak_bytes=max(r['gpu_phase_peaks']['whole_reserved_peak'] for r in values),
                decode_gpu_allocated_peak_bytes=max(r['gpu_phase_peaks']['decode_peak'] for r in values),
                final_gpu_allocated_bytes=[r['boundaries'][-1]['gpu_allocated_bytes'] for r in values],
                sampled_peak_rss_bytes=[max(s['VmRSS_bytes'] for s in r['rss_samples']) for r in values],
                host_cache_flush_rss_drop_bytes=[r['cleanup'][0]['VmRSS_bytes']-r['cleanup'][1]['VmRSS_bytes'] for r in values])
        summaries[geometry]=dict(root_requested_bytes=root,live_root_block_capacity_bytes=block,root_padding_bytes=block-root,
            exact_four_fresh_processes=True,arms=arms,
            whole_gpu_peak_reduction_pct=100*(1-arms['native_reuse']['whole_gpu_allocated_peak_bytes']/arms['fused_bounded']['whole_gpu_allocated_peak_bytes']),
            median_sampled_rss_change_bytes=statistics.median(arms['native_reuse']['sampled_peak_rss_bytes'])-statistics.median(arms['fused_bounded']['sampled_peak_rss_bytes']))
    prior=Path('results/hardware/reuse_visible_model_20261002/verified_parity_resource.json')
    known=json.loads(prior.read_text());assert known['verified']
    raw_resource=Path('results/hardware/reuse_visible_model_20261002/resource_shared.json')
    assert sha(raw_resource)==known['evidence_sha256']['resource_shared.json']
    data=json.loads(raw_resource.read_text())
    # The same single pinned root remains live throughout all policy phases.
    assert all(r['reserved_host_bytes']==151257088 for r in data['boundaries'])
    corrected=(known['known_policy_kv_peak_mib']*1048576)+268435456-151257088
    result=dict(verified=True,fresh_processes=8,preflight_cases=6,geometries=summaries,
        known_policy_kv_lower_bound_mib=known['known_policy_kv_peak_mib'],
        pinned_block_corrected_policy_kv_lower_bound_mib=corrected/1048576,
        pinned_block_corrected_512mib_gate_passed=corrected<=512*1048576,
        whole_model_10pct_memory_gate_passed=all(r['whole_gpu_peak_reduction_pct']>=10 for r in summaries.values()),
        direct_root_capacity_measurement=True,host_global_cached_total_measured=False,
        source_bundle_sha256=sha(BASE/'source.tar.gz'),known_resource_proof_sha256=sha(prior),
        evidence_sha256=evidence|{n:sha(BASE/n) for n in ('block_preflight.json','plan.json','commands.json','upstream_sources.json')},
        verifier_sha256=sha(Path(__file__)),limitations='Private ABI capacity read validated against installed/upstream headers and allocator deleter. Owned root only, excludes additional cached host blocks, GPU allocator rounding/workspaces/driver and other process memory. Adding root padding to the SAME prior known-KV trace gives a stronger lower bound, not complete actual memory equality. Fresh memory harness retains different forward outputs from legacy timing instrumentation; decode peaks not interchangeable. RSS samples lowerbound, VmHWM not phase-local, cleanup includes diagnostic canonical hashing. No speed/quality/physical-STT claim.')
    path=BASE/'verified_report.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
