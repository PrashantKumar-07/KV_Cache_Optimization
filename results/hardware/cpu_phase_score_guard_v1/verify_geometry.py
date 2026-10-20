"""Current-source observed geometry, separate from the passing memory envelope."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=BASE/'verified_geometry.json';assert not out.exists()
p=read(BASE/'resource_plan.json');r=read(BASE/'resource.json');proof=read(BASE/'verified_resource.json')
assert proof['verified'] and proof['resource_primary_passed']
for n,h in proof['evidence_sha256'].items():assert sha(BASE/n)==h,n
assert p['actual_driver_sha256']==sha(BASE/p['actual_driver'])==r['resource_append_driver_sha256']
old=read(ROOT/'results/hardware/cpu_phase_geometry_v1/plan.json');baseline=ROOT/old['baseline'];assert sha(baseline)==old['baseline_sha256']
s=read(baseline)
for n in ['input_ids_sha256','prompt_sha256','teacher_source_sha256','teacher_token_ids']:assert r[n]==s[n],n
own=r['ownership_boundaries'];width=r['forward_geometry'];snap=[x['cache'] for x in s['boundaries'] if 'cache' in x]
assert len(own)==len(snap)==51 and len(width)==50
for i,(a,z) in enumerate(zip(own,snap)):
 assert a['step']==i and a['distinct_owned_kv_storage_bytes']=={'cuda:0':134217728,'cpu':151257088}
 assert a['cpu_metadata_owner_bytes']==1116896 and a['gpu_metadata_owner_bytes']==417792 and a['cold_capacity_rows']==[1154]*32
 assert all(x['stored_visible_heads']==8 and x['visible_rows']==1023 and x['hot_rows']==991 and x['cold_rows']<=1090 for x in a['layers'])
 assert sum(z['distinct_owned_kv_storage_bytes'].values())==(974+i)*524288
for i,x in enumerate(width):
 assert x['input_rows']==1 and x['before_cache_rows']==[1023]*32 and x['stored_heads']==[8]*32
 assert s['forward_attended_rows_by_layer'][i]==[975+i]*32
result=dict(verified=True,current_source_resource_and_geometry_passed=True,candidate_KV_root_mib=272.25,candidate_KV_CPU_mib=144.25,candidate_KV_GPU_mib=128,candidate_scoped_active_plus_pinned_CPU_mib=proof['candidate_active_plus_cpu_array_bound_mib'],faithful_snap_KV_root_mib_range=[487,512],candidate_forward_rows=[1024,1024],faithful_snap_forward_rows=[975,1024],resource_mismatches=51,width_mismatches=49,equal_resource_P0_passed=False,full_goal_complete=False,ordinary_adoption=False,limits='Cache root bytes include unused backing capacity, distinct shared roots counted once. Complete peak includes actual256MiB rounded pinned block and CPU metadata separately. No timing or quality claim. Unowned/transient/native container and driver scopes unmeasured.',algebra='Constant272.25MiB vs faithful Snap0.5MiB/step cannot match across51 boundaries with any fixed integer prompt capacity; the current quarter-MiB candidate allocation also cannot equal a single half-MiB expanded Snap row boundary.',evidence_sha256={n:sha(BASE/n) for n in ['resource_plan.json','resource.json','verified_resource.json','audit_geometry_resources.py','verify_geometry.py']},baseline_evidence=dict(path=old['baseline'],sha256=old['baseline_sha256']))
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','baseline_evidence']},indent=2))
