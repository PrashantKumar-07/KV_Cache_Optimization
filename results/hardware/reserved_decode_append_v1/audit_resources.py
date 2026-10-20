"""Fresh16K50 ordinary caller with reservedappend/sharedhot ownership only."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import audit_bulk_resources as audit
from reserved_decode_append import ReservedAppendPolicy,AppendLease
BASE=Path(__file__).resolve().parent

def main():
    output=BASE/'resource.json';assert not output.exists()
    original=(audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,audit.fresh.atomic_json,audit.fresh.main,sys.argv)
    models=[];leases=[];final={}
    class Loader:
        @staticmethod
        def from_pretrained(*args,**options):
            model=original[0].from_pretrained(*args,**options);models.append(model);return model
    class ResourcePolicy(ReservedAppendPolicy):
        def reset(self):
            final.update(reserved_buffer_allocations=getattr(self,'reserved_buffer_allocations',0),shared_hot_prefixes=getattr(self,'shared_hot_prefixes',0),prototype_counts={f:sum(getattr(c,f,0) for c in self.layer_caches.values()) for f in ['host_read_bytes','host_write_bytes','host_initial_write_bytes']})
            final['prototype_counts'].update(native_h2d_calls=getattr(self.manager,'h2d_calls',0),native_d2h_calls=getattr(self.manager,'d2h_calls',0))
            super().reset()
    def factory(**options):
        policy=ResourcePolicy(**options);leases.append(AppendLease(models[0],policy));return policy
    def write(path,report):
        for lease in leases:lease.close()
        report.update(final,append_fast_updates=sum(v.fast_updates for v in leases),append_fallback_updates=sum(v.fallback_updates for v in leases),append_leases_closed=all(not v.handles and not v.pending for v in leases),resource_append_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),append_limits='Samefreshmeasurement/modelstorage/pinnedallocator boundaries; newGPUsparebuffer allocs charged viaTorchpeaks. No totalRSS/unownedhost/driver scope. Resourceprimary <=512MiB OR>=10%ordinary active decrease; failure stops standaloneexpansion.')
        original[2](path,report)
    def run():
        argv=sys.argv;sys.argv=[*argv,'--prefill-last-logits']
        try:original[3]()
        finally:sys.argv=argv
    audit.fresh.AutoModelForCausalLM=Loader;audit.BulkPromotionPolicy=factory;audit.fresh.atomic_json=write;audit.fresh.main=run
    sys.argv=['audit','--kind','fresh','--arm','bulk','--hot','991','--cold','1090','--output',str(output)]
    try:audit.main()
    finally:
        for lease in leases:lease.close()
        audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,audit.fresh.atomic_json,audit.fresh.main,sys.argv=original
        (BASE/'resource_cleanup.json').write_text(json.dumps(dict(append_leases_closed=all(not v.pending and not v.handles for v in leases)),indent=2)+'\n')
if __name__=='__main__':main()
