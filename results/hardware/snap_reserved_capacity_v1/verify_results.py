"""Fixed genuine baseline capacity, exact fidelity and complete resource limits."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    output=BASE/'verified_report.json';assert not output.exists()
    plan=read(BASE/'plan.json');m=read(BASE/'implementation.json')
    assert sha(BASE/'plan.json')==m['plan_sha256']
    assert sha(BASE/'implementation_source.tar.gz')==m['archive_sha256']
    assert sha(BASE/'control_source.tar.gz')==plan['archive_sha256']
    with tarfile.open(BASE/'implementation_source.tar.gz') as t:
        sources={p.name:hashlib.sha256(t.extractfile(p).read()).hexdigest() for p in t.getmembers() if p.isfile()}
    assert sources==m['sources'] and all(sha(ROOT/p)==v for p,v in sources.items())
    assert all(sources[p]==v for p,v in plan['ordinary_sources_sha256'].items())
    assert sha(BASE/'cpu.json')==m['cpu_sha256'] and sha(BASE/'errors.json')==m['errors_sha256']
    cpu=read(BASE/'cpu.json');errors=read(BASE/'errors.json')
    assert cpu['passed'] and len(cpu['pairs'])==8 and cpu['populated_kv_bits_exact'] and cpu['overflow_before_mutation']
    assert errors['passed'] and [r['roots_released'] for r in errors['rows']]==[2,6]
    prior=ROOT/'results/hardware/snap_actual_envelope_v1'
    assert sha(prior/'verified_report.json')==plan['control_proof_sha256']
    assert sha(ROOT/plan['control_raw'])==plan['control_raw_sha256']
    old=read(ROOT/plan['control_raw']);fresh=read(BASE/'fresh.json');paired=read(BASE/'gpu_preservation.json')
    jobs=read(BASE/'jobs.json');assert len(jobs)==2 and all(j['status']=='finished' and j['exit_code']==0 for j in jobs.values())
    assert fresh['passed'] and paired['passed']
    for d in (fresh,paired):
        for field in ('predicted_token_ids','teacher_token_ids','input_ids_sha256'):assert d[field]==old[field]
    for field in ('prompt_sha256','teacher_source_sha256','constant_model_storage_bytes','source_sha256'):
        assert fresh[field]==old[field]
    assert paired['frames_verified']==51 and len(paired['frames'])==51
    assert all(frame['bits_exact'] and frame['rows']==974+step for step,frame in enumerate(paired['frames']))
    assert all(frame['populated_bytes']==(974+step)*512*1024 for step,frame in enumerate(paired['frames']))
    assert all(frame.get('max_logit_absolute_delta',0)==0 for frame in paired['frames'])
    assert paired['roots_released']==64 and paired['hooks_groups_restored'] and paired['overflow_before_mutation']
    assert fresh['reserved_root_count']==fresh['reserved_roots_expired']==64
    assert fresh['adapter_hooks_and_groups_restored'] and fresh['owned_pinned_peak_bytes']==fresh['owned_pinned_after_cleanup_bytes']==0
    assert read(BASE/'fresh_cleanup.json')==dict(factory_writer_argv_restored=True,weak_handles_released=True)
    assert len(fresh['boundaries'])==51 and fresh['forward_attended_rows_by_layer']==old['forward_attended_rows_by_layer']
    for i,row in enumerate(fresh['boundaries']):
        assert row['cache']['distinct_owned_kv_storage_bytes']=={'cuda:0':512*2**20}
        assert all(layer['visible_rows']==974+i and layer['stored_visible_heads']==32 for layer in row['cache']['layers'])
    assert fresh['reserved_driver_sha256']==sources[str((BASE/'audit_fresh.py').relative_to(ROOT))]
    constant=fresh['constant_model_storage_bytes']
    envelope=lambda d:(d['gpu_phase_peaks']['decode_peak']-constant)/2**20
    reserve=lambda d:(d['gpu_phase_peaks']['whole_reserved_peak']-constant)/2**20
    report=dict(verified=True,fidelity_primary_passed=True,actual_reserved_kv_capacity_mib=512,
        populated_rows_range=[974,1024],boundary_frames=51,populated_kv_bits_exact=True,logits_exact=True,
        all50_frozen_predictions_replayed=True,capacity_overflow_before_mutation=True,roots_hooks_groups_released=True,
        original_active_decode_mib=envelope(old),reserved_variant_active_decode_mib=envelope(fresh),
        original_reserved_envelope_mib=reserve(old),reserved_variant_reserved_envelope_mib=reserve(fresh),
        original_prefill_peak_gb=old['gpu_phase_peaks']['prefill_peak']/1e9,
        reserved_variant_prefill_peak_gb=fresh['gpu_phase_peaks']['prefill_peak']/1e9,
        complete_512mib_active_passed=envelope(fresh)<=512,ordinary_source_changed=False,
        original_baseline_remains_control=True,ordinary_integration_ready=False,
        exact_tiered_resource_and_width_match=False,full_goal_complete=False,all_owned_jobs_finished=True,
        disposition='Research-only faithful selection baseline with genuine fixed future KV backing; no speed/memory-win claim.',
        next='Use measured allocator capacity, actual alias ownership and comparable attention geometry to preregister the TieredKV joint comparison; do not force equality with dummy allocation.',
        limits='One model/seed16K50. Every populatedKV bit and logits match in paired GPU run; fresh capacity root fixed512MiB. '
            'Full decode active exceeds512MiB; retained CUDA reservation grows. No timing/current independent quality, exact '
            'TieredKV usage equality, ordinary baseline replacement or physicalSTT claim.',
        evidence_sha256={p:sha(BASE/p) for p in ('plan.json','control_source.tar.gz','implementation.json','implementation_source.tar.gz',
            'cpu.json','errors.json','gpu_preservation.json','fresh.json','fresh_cleanup.json','jobs.json')},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence_sha256'},indent=2))


if __name__=='__main__':main()
