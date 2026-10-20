"""Fresh16K50 ordinary caller with reservedappend/sharedhot ownership only."""
import hashlib,json,sys
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from audit_snap_tier_resources import snapshot
import audit_bulk_resources as audit
from cold_boundary_policy import ColdBoundaryPolicy as ReservedAppendPolicy
from shared_decode_transition import SharedAppendLease as AppendLease
BASE=Path(__file__).resolve().parent

def main():
    output=BASE/'cpu.json';assert not output.exists()
    original=(audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,audit.fresh.atomic_json,audit.fresh.main,sys.argv)
    models=[];leases=[];final={};ownership=[];widths=[];handles=[];baselines=[]
    def before(model,args,options):
        ids=options.get("input_ids",args[0] if args else None);past=options.get("past_key_values")
        if past is not None:
            widths.append(dict(input_rows=ids.shape[1],before_cache_rows=[l.keys.shape[-2] for l in past.layers],stored_heads=[l.keys.shape[1] for l in past.layers]))
    class Loader:
        @staticmethod
        def from_pretrained(*args,**options):
            model=original[0].from_pretrained(*args,**options);models.append(model);baselines.append((len(model._forward_pre_hooks),len(model._forward_hooks)));handles.append(model.register_forward_pre_hook(before,with_kwargs=True));return model
    class ResourcePolicy(ReservedAppendPolicy):
        def _visible_tuple(self,past):
            result=super()._visible_tuple(past)
            visible=SimpleNamespace(layers=[SimpleNamespace(keys=k,values=v) for k,v in result])
            row=snapshot(visible,self,len(ownership))
            row.update(cpu_metadata_owner_bytes=self.cpu_arena.owned_bytes if self.cpu_arena else 0,gpu_metadata_owner_bytes=self.state_arena.owned_gpu_bytes if self.state_arena else 0,cold_capacity_rows=[c.capacity for c in self.layer_caches.values()])
            ownership.append(row)
            return result
        def reset(self):
            final.update(reserved_buffer_allocations=getattr(self,'reserved_buffer_allocations',0),shared_hot_prefixes=getattr(self,'shared_hot_prefixes',0),shared_transition_bindings=getattr(self,'shared_transition_bindings',0),shared_immediate_compactions=sum(getattr(c,'shared_immediate_compactions',0) for c in self.layer_caches.values()),batched_deferred_compactions=sum(getattr(c,'batched_deferred_compactions',0) for c in self.layer_caches.values()),bounded_native_groups=getattr(self,'bounded_native_groups',0),bounded_scratch_peak_bytes=getattr(self,'bounded_scratch_peak_bytes',0),controller_plan_calls=getattr(self,'controller_plan_calls',0),controller_fallback_plans=getattr(self,'controller_fallback_plans',0),controller_rank_groups=getattr(self,'controller_rank_groups',0),controller_rank_h2d_bytes=getattr(self,'controller_rank_h2d_bytes',0),controller_cpu_streak_array_peak=getattr(self,'controller_cpu_streak_array_peak',0),controller_backup_calls=sum(getattr(c,'controller_backup_calls',0) for c in self.layer_caches.values()),controller_address_calls=getattr(self.manager,'controller_address_calls',0),controller_address_array_peak=getattr(self.manager,'controller_address_array_peak',0),prototype_counts={f:sum(getattr(c,f,0) for c in self.layer_caches.values()) for f in ['host_read_bytes','host_write_bytes','host_initial_write_bytes']})
            final['prototype_counts'].update(native_h2d_calls=getattr(self.manager,'h2d_calls',0),native_d2h_calls=getattr(self.manager,'d2h_calls',0))
            arena=getattr(self,'state_arena',None)
            final.update(pooled_prepare_calls=getattr(self,'pooled_prepare_calls',0),pooled_compact_groups=getattr(self,'pooled_compact_groups',0),pooled_owned_gpu_bytes=arena.owned_gpu_bytes if arena else 0,pooled_descriptor_h2d_bytes=arena.descriptor_h2d_bytes if arena else 0,pooled_scratch_peak_bytes=arena.scratch_peak_bytes if arena else 0)
            cpu=getattr(self,"cpu_arena",None)
            final.update(cold_boundary_steps=self.cold_boundary_steps,cold_boundary_fallback_calls=self.cold_boundary_fallback_calls,cold_boundary_descriptor_h2d_bytes=self.cold_boundary_descriptor_h2d_bytes,cold_boundary_rank_peak_bytes=self.cold_boundary_rank_peak_bytes,cold_boundary_cpu_peak_bytes=self.cold_boundary_cpu_peak_bytes,bulk_payload_steps=self.bulk_payload_steps,bulk_payload_fallback_steps=self.bulk_payload_fallback_steps,bulk_descriptor_h2d_bytes=self.bulk_descriptor_h2d_bytes,bulk_scratch_peak_bytes=self.bulk_scratch_peak_bytes,bulk_paid_peak_bytes=self.bulk_paid_peak_bytes,bulk_dispatches=getattr(self.manager,'bulk_dispatches',0),cpu_promote_calls=self.cpu_promote_calls,cpu_append_calls=self.cpu_append_calls,cpu_demote_calls=self.cpu_demote_calls,cpu_fallback_steps=self.cpu_fallback_steps,cpu_evictions=sum(c.cpu_evictions for c in self.layer_caches.values()),cpu_reclaims=sum(c.cpu_reclaims for c in self.layer_caches.values()),cpu_owned_bytes=cpu.owned_bytes if cpu else 0)
            super().reset()
    def factory(**options):
        policy=ResourcePolicy(**options);leases.append(AppendLease(models[0],policy));return policy
    def write(path,report):
        for lease in leases:lease.close()
        report.update(ownership_boundaries=ownership,forward_geometry=widths,**final,append_fast_updates=sum(v.fast_updates for v in leases),append_fallback_updates=sum(v.fallback_updates for v in leases),append_leases_closed=all(not v.handles and not v.pending for v in leases),resource_append_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),append_limits='Samefreshmeasurement/modelstorage/pinnedallocator boundaries; newGPUsparebuffer allocs charged viaTorchpeaks. No totalRSS/unownedhost/driver scope. Resourceprimary <=512MiB OR>=10%ordinary active decrease; failure stops standaloneexpansion.')
        original[2](path,report)
    def run():
        argv=sys.argv;sys.argv=[*argv,'--prefill-last-logits']
        try:original[3]()
        finally:sys.argv=argv
    audit.fresh.AutoModelForCausalLM=Loader;audit.BulkPromotionPolicy=factory;audit.fresh.atomic_json=write;audit.fresh.main=run
    sys.argv=['audit','--kind','fresh','--arm','bulk','--hot','649','--cold','1982','--output',str(output)]
    try:audit.main()
    finally:
        for lease in leases:lease.close()
        for h in handles:h.remove()
        assert not models or baselines[0]==(len(models[0]._forward_pre_hooks),len(models[0]._forward_hooks))
        audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,audit.fresh.atomic_json,audit.fresh.main,sys.argv=original
        (BASE/'resource_cleanup.json').write_text(json.dumps(dict(append_leases_closed=all(not v.pending and not v.handles for v in leases)),indent=2)+'\n')
if __name__=='__main__':main()
