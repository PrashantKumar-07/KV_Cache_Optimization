"""Check public CUDA root-range measurements against frozen fresh-process data."""
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((BASE/'plan.json').read_text())
    assert sha(BASE/'source.tar.gz')==plan['source_bundle_sha256']
    assert sha(BASE.parent/'public_preflight.json')==plan['public_preflight_sha256']
    assert sha(BASE.parent/'verified_report.json')==plan['private_audit_sha256']
    pre=json.loads((BASE.parent/'public_preflight.json').read_text());assert pre['passed'] and pre['nonpinned_rejection']
    assert len(pre['checks'])==6 and all(r['private_capacity_exact'] and r['view_storage_alias_exact'] for r in pre['checks'])
    hashes={}
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile():hashes[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert hashes==json.loads((BASE/'source_manifest.json').read_text())
    assert 'experiments/native_host_block.cpp' not in hashes
    commands=json.loads((BASE/'commands.json').read_text());assert len(commands)==4 and all(r['returncode']==0 and not r['ninja_on_path'] for r in commands)
    summaries={};evidence={}
    for geometry,root,block in (('shared',151257088,268435456),('default',276824064,536870912)):
        rows={}
        for arm in ('fused_bounded','native_reuse'):
            name=f'{geometry}_{arm}.json';raw=json.loads((BASE/name).read_text());old=json.loads((BASE.parent/f'{geometry}_0_{arm}.json').read_text())
            assert raw['passed'] and raw['actual_arm']==arm
            assert raw['host_block_probe_method']=='public_cuPointerGetAttribute' and raw['host_block_probe_sha256']==pre['probe_sha256']
            assert all(hashes[p]==value for p,value in raw['actual_source_sha256'].items())
            for field in ('state_sha256','predicted_token_ids','counters','occupied_host_bytes','reserved_host_bytes','prompt_sha256',
                'teacher_source_sha256','teacher_token_ids','input_ids_sha256','gpu_phase_peaks'):
                assert raw[field]==old[field],(name,field)
            assert raw['owned_pinned_peak_bytes']==root and raw['owned_host_allocator_block_peak_bytes']==block
            assert raw['owned_host_allocator_block_after_cleanup_bytes']==raw['owned_pinned_after_cleanup_bytes']==0
            assert len(raw['host_block_observations'])==1
            observed=raw['host_block_observations'][0]
            assert observed['requested_storage_bytes']==root and observed['allocator_block_bytes']==block and observed['public_cuda_memory_type']==1
            assert len(raw['boundaries'])==52 and len(raw['predicted_token_ids'])==50
            rows[arm]=dict(whole_gpu_peak_bytes=raw['gpu_phase_peaks']['whole_allocated_peak'],decode_gpu_peak_bytes=raw['gpu_phase_peaks']['decode_peak'],
                final_gpu_allocated_bytes=raw['boundaries'][-1]['gpu_allocated_bytes'],sampled_peak_rss_bytes=max(r['VmRSS_bytes'] for r in raw['rss_samples']))
            evidence[name]=sha(BASE/name)
        assert rows['fused_bounded']['whole_gpu_peak_bytes']==rows['native_reuse']['whole_gpu_peak_bytes']
        summaries[geometry]=dict(root_request_bytes=root,public_cuda_host_range_bytes=block,arms=rows)
    private=json.loads((BASE.parent/'verified_report.json').read_text())
    assert private['pinned_block_corrected_policy_kv_lower_bound_mib']==518.00390625 and not private['pinned_block_corrected_512mib_gate_passed']
    result=dict(verified=True,fresh_processes=4,public_private_preflight_matches=6,no_host_probe_extension_or_ninja_needed=True,
        geometries=summaries,pinned_range_corrected_policy_kv_lower_bound_mib=518.00390625,
        shared_512mib_gate_passed=False,whole_model_10pct_memory_gate_passed=False,
        private_audit_sha256=sha(BASE.parent/'verified_report.json'),source_bundle_sha256=sha(BASE/'source.tar.gz'),
        evidence_sha256=evidence,verifier_sha256=sha(Path(__file__)),
        limitations='Public CUDA host allocation address ranges match the archived private HostBlock capacities; active code no longer reads private layouts or compiles a host extension. Ranges do not measure OS page rounding, other cached allocations, complete GPU/driver/process memory or physical STT. Stronger lower bound already fails512MiB. Fresh harness lifetime differs from legacy profile. No speed/quality/superiority claim.')
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
