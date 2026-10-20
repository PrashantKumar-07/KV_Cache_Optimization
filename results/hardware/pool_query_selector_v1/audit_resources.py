"""Fresh ordinary resource caller with scoped native-pool current-query hooks."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import audit_bulk_resources as audit
from bulk_promotion_policy import BulkPromotionPolicy
from pool_query_selector import PoolQuerySelector


def main():
    base=Path(__file__).resolve().parent;output=base/'resource.json';assert not output.exists()
    original=(audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,
              audit.fresh.atomic_json,audit.fresh.main,sys.argv)
    models=[];selectors=[];before=[];final={}
    class Loader:
        @staticmethod
        def from_pretrained(*args,**options):
            model=original[0].from_pretrained(*args,**options)
            models.append(model)
            before.extend(len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers)
            return model
    class QueryPolicy(BulkPromotionPolicy):
        def __init__(self,**options):
            super().__init__(**options)
            self.selector=PoolQuerySelector(models[0],self);selectors.append(self.selector)
        def reset(self):
            final.update(query_selected_layers=self.selector.selected_layers,
                query_scanned_rows=self.selector.scanned_rows,
                prototype_counts={name:sum(getattr(c,name,0) for c in self.layer_caches.values())
                    for name in ('host_read_bytes','host_write_bytes','host_initial_write_bytes')},
                native_h2d_calls=getattr(self.manager,'h2d_calls',0),
                native_d2h_calls=getattr(self.manager,'d2h_calls',0))
            self.selector.close();super().reset()
    def write(path,report):
        removed=bool(models) and before==[len(layer.self_attn._forward_pre_hooks) for layer in models[0].model.layers]
        report.update(final,query_hooks_removed=removed,query_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            query_limits='Unchanged fresh16K50 teacher measurement/reset boundaries; full native paired cold scan. '
                'Transfer costs charged. No quality/timing/resource equality/physicalSTT/ordinary adoption proof.')
        if report['passed']:assert removed and all(not s.handles for s in selectors)
        original[2](path,report)
    def run():
        saved=sys.argv;sys.argv=[*saved,'--prefill-last-logits']
        try:original[3]()
        finally:sys.argv=saved
    audit.fresh.AutoModelForCausalLM=Loader;audit.BulkPromotionPolicy=QueryPolicy
    audit.fresh.atomic_json=write;audit.fresh.main=run
    sys.argv=['audit','--kind','fresh','--arm','bulk','--hot','991','--cold','1090','--output',str(output)]
    try:audit.main()
    finally:
        for selector in selectors:selector.close()
        (audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,
         audit.fresh.atomic_json,audit.fresh.main,sys.argv)=original
        cleanup=base/'resource_cleanup.json';assert not cleanup.exists()
        cleanup.write_text(json.dumps(dict(selectors_closed=all(not s.handles for s in selectors),
            attention_pre_hooks_restored=bool(models) and before==[
                len(layer.self_attn._forward_pre_hooks) for layer in models[0].model.layers]),indent=2)+'\n')
if __name__=='__main__':main()
