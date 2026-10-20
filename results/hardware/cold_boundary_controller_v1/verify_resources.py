"""Read-only checks of frozen integrated bulk evidence and scoped memory gate."""
import hashlib,json,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
plan=read(B/'plan.json');control=R/plan['control'];assert sha(control)==plan['control_sha256']
a=read(control);c=read(B/'cpu.json');assert a['passed'] and c['passed']
for kind in ['gpu','parity','failure','resource']:
 m=read(B/(kind+'_implementation.json'));archive=B/(kind+'_source.tar.gz');assert sha(archive)==m['archive_sha256']
 with tarfile.open(archive) as t:
  for path,h in m['sources'].items():
   assert sha(R/path)==h,path
   assert hashlib.sha256(t.extractfile(path).read()).hexdigest()==h,path
 assert m['plan_sha256']==sha(B/('plan.json' if kind=='gpu' else kind+'_plan.json'))
for name in ['tests','gpu','parity_4096_12','parity_16384_50','failure','resource']:
 j=read(B/(name+'_job.json'));assert j['status']=='finished' and j['exit_code']==0
assert read(B/'tests_job.json')['suite_passed']==518
x=read(B/'gpu.json');assert x['passed'] and x['hooks_leases_and_cache_owners_released']
assert x['control']['frames']==x['candidate']['frames'];assert x['partial_queue_failure']['queued_before_failure']
assert x['partial_queue_failure']['pinned_after_cleanup_bytes']==0 and x['partial_queue_failure']['hooks_leases_policy_released']
for name,n in [('parity_4096_12',12),('parity_16384_50',50)]:
 x=read(B/(name+'.json'));assert x['passed'] and x['owned_leases_closed']
 assert x['rows'][1]['phase_counts']['bulk_steps']==n
 assert x['rows'][1]['phase_counts']['cold_steps']==n and x['rows'][1]['phase_counts']['cold_fallback_calls']==1
for k in ['state_sha256','predicted_token_ids','counters','prototype_counts','forward_geometry']:assert c[k]==a[k],k
assert c['bulk_payload_steps']==50 and c['bulk_payload_fallback_steps']==0 and c['bulk_dispatches']==25
assert len(c['ownership_boundaries'])==51
assert all(sum(x['distinct_owned_kv_storage_bytes'].values())==341*2**20 for x in c['ownership_boundaries'])
model=c['resource_model']['constant_model_storage_bytes'];assert model==a['resource_model']['constant_model_storage_bytes']==14483464704
active=c['gpu_phase_peaks']['decode_peak']-model;old=a['gpu_phase_peaks']['decode_peak']-model
gpu_change=100*(active/old-1);reserve_change=100*(c['gpu_phase_peaks']['whole_reserved_peak']/a['gpu_phase_peaks']['whole_reserved_peak']-1)
assert gpu_change<=2 and reserve_change<=2
assert c['gpu_phase_peaks']['prefill_peak']/a['gpu_phase_peaks']['prefill_peak']<=1.10
assert c['owned_pinned_after_cleanup_bytes']==c['owned_host_allocator_block_after_cleanup_bytes']==0
assert c['append_leases_closed'] and read(B/'resource_cleanup.json')['append_leases_closed']
# Descriptor and converted int64 index arrays each fit the descriptor byte count.
# Twice the sum over calls conservatively bounds their peak; it is not a measured peak.
scratch_bound=2*c['bulk_descriptor_h2d_bytes']
cold_scratch_bound=2*c['cold_boundary_descriptor_h2d_bytes']
assert c['cold_boundary_steps']==50 and c['cold_boundary_fallback_calls']==1
f=read(B/'failure.json');assert f['passed'] and f['h2d_submitted_before_failure'] and f['pinned_after_cleanup_bytes']==0 and f['all_owned_pool_hooks_leases_released']
parts=dict(gpu_active_excluding_constant_model=active,owned_pinned_allocator_block=c['owned_host_allocator_block_peak_bytes'],canonical_cpu_arrays=c['cpu_owned_bytes'],controller_streak_array=c['controller_cpu_streak_array_peak'],controller_address_array=c['controller_address_array_peak'],new_cpu_descriptor_index_scratch_upper_bound=scratch_bound,cold_cpu_descriptor_index_scratch_upper_bound=cold_scratch_bound)
total=sum(parts.values());assert total<=512*2**20
out=B/'verified_resources.json';assert not out.exists()
out.write_text(json.dumps(dict(passed=True,scope_parts_bytes=parts,scope_mib=total/2**20,strict_gpu_decode_change_percent=gpu_change,reserved_change_percent=reserve_change,prefill_guard_passed=True,exact_replay_geometry_transfers=True,tests_passed=518,physical_queue_failure_cleanup=True,late_cold_h2d_failure_cleanup=True,evidence_sha256={str(p.relative_to(R)):sha(p) for p in [B/'plan.json',B/'cpu.json',control,B/'failure.json',B/'gpu.json',B/'parity_4096_12.json',B/'parity_16384_50.json']},limits='Scoped allocation plus owned CPU arrays and conservative new CPU scratch bound. Unowned Python/native container, driver and process host allocations are outside scope. Prefill unchanged; faithful Snap prefill comparison remains failed, trajectory equality false. No adoption or full goal completion.',full_goal_complete=False),indent=2)+'\n')
print(out,total/2**20,gpu_change)
