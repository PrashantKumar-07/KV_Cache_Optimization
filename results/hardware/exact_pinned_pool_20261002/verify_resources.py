"""Recompute whole-decode resource envelopes from preserved raw allocation data."""
import hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifests={}
    for filename in ('source_model_v2.tar.gz','integrity_resource_source.tar.gz'):
        with tarfile.open(B/filename) as archive:
            for member in archive.getmembers():
                if member.isfile():manifests[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    plan=json.loads((B/'resources_plan.json').read_text())
    assert plan['driver_sha256']==manifests['experiments/audit_exact_pool_resources.py']
    results={}
    paths=[]
    for name,hot,cold in [('shared',991,1090),('default',1024,2048)]:
        path=B/f'resources_{name}.json';paths.append(path.name)
        raw=json.loads(path.read_text());assert raw['passed'] and not raw.get('error')
        assert raw['config']['hot']==hot and raw['config']['cold']==cold
        assert raw['config']['contexts']==[16384] and raw['config']['decode']==50 and raw['config']['blocks']==0
        assert len(raw['rows'])==2
        assert raw['arm_mapping']=={'fused_bounded':'cached_native_reuse','native_reuse':'exact_native_reuse'}
        assert all(manifests[p]==v for p,v in raw['exact_source_sha256'].items())
        stats=raw['exact_allocation_stats'];assert stats['live_allocations']==stats['live_requested_bytes']==0
        events=stats['events'];assert len(events)==2 and events[0]['kind']=='allocate' and events[1]['kind']=='release'
        assert events[0]['bytes']==events[1]['bytes']==(cold+64)*131072
        arms={}
        reference=raw['rows'][0]
        for row in raw['rows']:
            for key in ('state_sha256','history_sha256','every_visible_state_sha256','explicit_h2d_payload_bytes',
                        'explicit_d2h_payload_bytes','initial_d2h_payload_bytes','occupied_host_bytes',
                        'reserved_host_bytes','native_d2h_calls','native_h2d_calls'):
                assert row[key]==reference[key],(name,key)
            for key in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):
                assert row['trials'][0][key]==reference['trials'][0][key]
            item=row['whole_decode_resources'];root=item['root'];weights=item['constant_model_storage_bytes']
            assert item['driver_sha256']==plan['driver_sha256'] and weights==14483464704
            assert root['requested_storage_bytes']==(cold+64)*131072
            assert item['gpu_allocated_peak_bytes']==round(row['peak_memory_gb']*1e9)
            assert item['gpu_reserved_peak_bytes']>=item['gpu_allocated_peak_bytes']>=weights
            active=item['gpu_allocated_peak_bytes']-weights+root['allocator_block_bytes']
            reserved=item['gpu_reserved_peak_bytes']-weights+root['allocator_block_bytes']
            assert active==item['allocated_extra_plus_host_range_bytes'] and reserved==item['reserved_extra_plus_host_range_bytes']
            ceiling=512*2**20;assert item['ceiling_bytes']==ceiling
            arms[raw['arm_mapping'][row['arm']]]=dict(allocated_extra_plus_host_range_bytes=active,
                allocated_extra_plus_host_range_mib=active/2**20,reserved_extra_plus_host_range_bytes=reserved,
                reserved_extra_plus_host_range_mib=reserved/2**20,allocated_scope_ceiling_pass=active<=ceiling,
                reserved_scope_ceiling_pass=reserved<=ceiling)
        exact=arms['exact_native_reuse']
        assert exact['allocated_scope_ceiling_pass']==(name=='shared') and not exact['reserved_scope_ceiling_pass']
        results[name]=arms
    report=dict(verified=True,geometries=results,actual_resource_equality_proven=False,
        candidate_ready_for_opt_in_cli=False,decision='Keep research-only: usual active envelope fails; reserved scope fails both; remaining external allocation scopes unmeasured.',
        limitations='Legacy caller frees prefill logits before decode. Torch allocation includes forward/policy temporaries and metadata. Reserved includes retained prefill blocks; it is reported separately. Excludes non-Torch CUDA-driver and other host allocations/cache. Root maximum and GPU maximum form a conservative envelope; no timing/model-wide memory/physical STT/SnapKV superiority claim.',
        evidence_sha256={p:sha(B/p) for p in paths+['resources_plan.json','source_model_v2.tar.gz','integrity_resource_source.tar.gz']},verifier_sha256=sha(Path(__file__)))
    output=B/'verified_resources.json';assert not output.exists();output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
