"""Existing recorded4K12 fullmodel pair with scoped reservedappend cache lease."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import compare_reuse_visible as driver
from compare_bulk_promotion import RecordedBulkPolicy
from native_controller_policy import NativeControllerPolicy as ReservedAppendPolicy
from shared_decode_transition import SharedAppendLease as AppendLease
BASE=Path(__file__).resolve().parent
class RecordedAppendPolicy(driver.RecordedNativePolicy,ReservedAppendPolicy):
    append_counts={}
    def reset(self):
        type(self).append_counts=dict(reserved_buffer_allocations=getattr(self,'reserved_buffer_allocations',0),shared_hot_prefixes=getattr(self,'shared_hot_prefixes',0),shared_transition_bindings=getattr(self,'shared_transition_bindings',0),shared_immediate_compactions=sum(getattr(c,'shared_immediate_compactions',0) for c in self.layer_caches.values()),batched_deferred_compactions=sum(getattr(c,'batched_deferred_compactions',0) for c in self.layer_caches.values()),bounded_native_groups=getattr(self,'bounded_native_groups',0),bounded_scratch_peak_bytes=getattr(self,'bounded_scratch_peak_bytes',0),controller_plan_calls=getattr(self,'controller_plan_calls',0),controller_fallback_plans=getattr(self,'controller_fallback_plans',0),controller_rank_groups=getattr(self,'controller_rank_groups',0),controller_rank_h2d_bytes=getattr(self,'controller_rank_h2d_bytes',0),controller_cpu_streak_array_peak=getattr(self,'controller_cpu_streak_array_peak',0),controller_backup_calls=sum(getattr(c,'controller_backup_calls',0) for c in self.layer_caches.values()),controller_address_calls=getattr(self.manager,'controller_address_calls',0),controller_address_array_peak=getattr(self.manager,'controller_address_array_peak',0))
        super().reset()
def main():
    saved=(driver.RecordedNativePolicy,driver.trial.FusedBoundedPolicy,driver.trial.measure,driver.atomic_json,sys.argv)
    leases=[];summaries=[]
    def measure(model,*args,**kwargs):
        chosen=driver.trial.FusedBoundedPolicy;own=[]
        before=(len(model._forward_pre_hooks),len(model._forward_hooks))
        if chosen is RecordedAppendPolicy:
            def factory(**options):
                policy=chosen(**options);lease=AppendLease(model,policy);own.append(lease);leases.append(lease);return policy
            driver.trial.FusedBoundedPolicy=factory
        try:
            result=saved[2](model,*args,**kwargs)
            if own:
                result['append_lease_counts']=dict(fast_updates=sum(v.fast_updates for v in own),fallback_updates=sum(v.fallback_updates for v in own),pending_released=all(not v.pending for v in own),**RecordedAppendPolicy.append_counts)
                summaries.append(result['append_lease_counts'])
            return result
        finally:
            for lease in own:lease.close()
            assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
            driver.trial.FusedBoundedPolicy=chosen
    def write(path,report):
        report.update(actual_control='ordinary_bulk',actual_candidate='native_controller',append_summaries=summaries,owned_leases_closed=all(not v.pending and not v.handles for v in leases),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='Existingrecorded4K12 pair. Exactcache/pred/visibility/counters/nativeKV; no separatelycapturedlogits, memoryprimary or speed/quality/full512MiB/adoption claim.')
        saved[3](path,report)
    driver.RecordedNativePolicy=RecordedAppendPolicy;driver.trial.FusedBoundedPolicy=RecordedBulkPolicy;driver.trial.measure=measure;driver.atomic_json=write
    sys.argv=['check','--output',str(BASE/'gpu.json'),'--contexts','4096','--decode','12']
    try:driver.main()
    finally:
        for lease in leases:lease.close()
        driver.RecordedNativePolicy,driver.trial.FusedBoundedPolicy,driver.trial.measure,driver.atomic_json,sys.argv=saved
if __name__=='__main__':main()
