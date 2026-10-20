"""Fresh unchanged full-query resource caller with GPU native read-index maps."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import audit_bulk_resources as audit
from query_channel_reads import ChannelQueryPolicy as BulkPromotionPolicy, kernel
from query_channel_reads import ChannelQuerySelector as PoolQuerySelector


def main():
    base=Path(__file__).resolve().parent;output=base/'resource.json';assert not output.exists()
    native=json.loads((base/'native_v2.json').read_text());assert native['passed']
    assert hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest()==native['binary_sha256']
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
            final.update(visible_suffix_updates=self.selector.visible_suffix_updates,channel_key_reads=self.query_reader.key_reads,channel_value_reads=self.query_reader.value_reads,
                channel_key_rows=self.query_reader.key_rows,channel_value_rows=self.query_reader.value_rows,
                channel_native_binary_sha256=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest(),query_selected_layers=self.selector.selected_layers,
                query_scanned_rows=self.selector.scanned_rows,
                index_metadata_h2d_bytes=self.query_reader.metadata_h2d_bytes,index_refreshes=self.query_reader.refreshes,
                index_reads=self.query_reader.reads,index_gpu_storage_bytes=sum(t.numel()*t.element_size() for t in (self.query_reader.blocks,self.query_reader.live) if t is not None),
                prototype_counts={name:sum(getattr(c,name,0) for c in self.layer_caches.values())
                    for name in ('host_read_bytes','host_write_bytes','host_initial_write_bytes')},
                native_h2d_calls=getattr(self.manager,'h2d_calls',0),
                native_d2h_calls=getattr(self.manager,'d2h_calls',0))
            self.selector.close();super().reset()
            assert self.query_reader.blocks is None and self.query_reader.live is None
            final['index_metadata_released']=True
    def write(path,report):
        removed=bool(models) and before==[len(layer.self_attn._forward_pre_hooks) for layer in models[0].model.layers]
        report.update(final,query_hooks_removed=removed,query_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            query_limits='Unchanged fresh16K50 teacher measurement/reset boundaries; full native paired cold scan. '
                'KVpayload and candidate index-metadata copies charged separately; old index copies not measured. No quality/timing/resource equality/physicalSTT/ordinary adoption proof.')
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
