"""Verify fresh candidate ownership vectors against immutable faithful Snap data."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=BASE/'verified_geometry.json';assert not out.exists()
p=read(BASE/'plan.json');f=read(BASE/'gpu_implementation.json');r=read(BASE/'resource.json')
assert sha(BASE/'plan.json')==f['plan_sha256'] and sha(BASE/'gpu_source.tar.gz')==f['archive_sha256']
with tarfile.open(BASE/'gpu_source.tar.gz') as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
assert sources==f['sources'] and all(sha(ROOT/n)==h for n,h in sources.items())
assert read(BASE/'gpu_job.json')['exit_code']==0 and r['passed'] and r['append_leases_closed'] and read(BASE/'resource_cleanup.json')['append_leases_closed']
assert sha(ROOT/p['reference'])==p['reference_sha256'] and sha(ROOT/p['baseline'])==p['baseline_sha256']
ref=read(ROOT/p['reference']);snap=read(ROOT/p['baseline'])
for n in ['input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids','predicted_token_ids','state_sha256','counters','prototype_counts','gpu_phase_peaks','owned_host_allocator_block_peak_bytes']:assert r[n]==ref[n],n
for n in ['input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids']:assert r[n]==snap[n],n
for n,v in dict(cpu_promote_calls=50,cpu_append_calls=50,cpu_demote_calls=50,cpu_fallback_steps=0,cpu_evictions=1600,append_fast_updates=1600,append_fallback_updates=0).items():assert r[n]==v,n
assert r['owned_pinned_after_cleanup_bytes']==r['owned_host_allocator_block_after_cleanup_bytes']==0
own=r['ownership_boundaries'];width=r['forward_geometry'];baseline=[x['cache'] for x in snap['boundaries'] if 'cache' in x]
assert len(own)==len(baseline)==51 and len(width)==50
cpu_bytes=[];snap_bytes=[];cpu_width=[];snap_width=snap['forward_attended_rows_by_layer']
for i,(a,b) in enumerate(zip(own,baseline)):
 assert a['step']==i and len(a['layers'])==32
 assert a['distinct_owned_kv_storage_bytes']=={'cuda:0':134217728,'cpu':151257088}
 assert a['cpu_metadata_owner_bytes']==1116896 and a['gpu_metadata_owner_bytes']==417792
 assert a['cold_capacity_rows']==[1154]*32
 assert all(x['stored_visible_heads']==8 and x['visible_rows']==1023 and x['hot_rows']==991 and x['cold_rows']<=1090 for x in a['layers'])
 cpu_bytes.append(sum(a['distinct_owned_kv_storage_bytes'].values()));snap_bytes.append(sum(b['distinct_owned_kv_storage_bytes'].values()))
 assert snap_bytes[-1]==(974+i)*524288
for x in width:
 assert x['input_rows']==1 and x['before_cache_rows']==[1023]*32 and x['stored_heads']==[8]*32
 cpu_width.append([n+x['input_rows'] for n in x['before_cache_rows']])
assert snap_width==[[975+i]*32 for i in range(50)]
resource_equal=cpu_bytes==snap_bytes;width_equal=cpu_width==snap_width
assert not resource_equal and not width_equal
constant=ref['resource_model']['constant_model_storage_bytes'];active=(r['gpu_phase_peaks']['decode_peak']-constant+r['owned_host_allocator_block_peak_bytes']+r['cpu_owned_bytes']+r['controller_cpu_streak_array_peak']+r['controller_address_array_peak'])/2**20
result=dict(verified=True,prediction_state_counters_native_KV_and_phase_peaks_replayed=True,owned_hooks_pinned_roots_released=True,KV_boundaries=51,forward_boundaries=50,candidate_distinct_GPU_KV_mib=128,candidate_distinct_CPU_KV_mib=144.25,candidate_total_resident_KV_root_mib=272.25,candidate_scoped_active_plus_pinned_CPU_mib=active,faithful_snap_resident_KV_mib_range=[min(snap_bytes)/2**20,max(snap_bytes)/2**20],candidate_forward_width_range=[1024,1024],faithful_snap_forward_width_range=[975,1024],unequal_resource_boundaries=sum(a!=b for a,b in zip(cpu_bytes,snap_bytes)),unequal_width_steps=sum(a!=b for a,b in zip(cpu_width,snap_width)),equal_actual_resident_KV=False,equal_forward_width=False,matched_resource_P0_complete=False,full_goal_complete=False,ordinary_adoption=False,
 algebraic_finding='For fixed-capacity candidate roots, resident KV is constant272.25MiB. Faithful SnapKV grows0.5MiB/step; no single integer prompt capacity can match this vector across51 boundaries. Its row granularity0.5MiB also cannot equal272.25MiB at one boundary. Attended model heads remain32 in both; stored KV heads8 vs32 are different storage semantics.',
 next='Do not launch a claimed equal-resource quality/latency comparison with these unchanged fixed capacities. A legitimate ownership/retention comparison design must account for growing faithful baseline and grouped-head storage; report physical allocations and attention geometry separately without padding or modifying the faithful baseline.',limits=p['limits'],evidence_sha256={n:sha(BASE/n) for n in ['plan.json','gpu_implementation.json','gpu_source.tar.gz','resource.json','resource_cleanup.json','gpu_job.json','gpu.log','audit_resources.py','verify_geometry.py']},external_evidence_sha256={p['reference']:p['reference_sha256'],p['baseline']:p['baseline_sha256']})
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','external_evidence_sha256']},indent=2))
