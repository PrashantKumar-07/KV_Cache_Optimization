"""Current upstream KV-head SnapKV selector, opt-in Transformers5 bridge.

Reference: Zefan-Cai/KVCache-Factory commit
68cd9551a63fedf362ddb0edb008badd04c24996. Grouped-query head scores are
averaged before pooling/top-k, retaining eight actual KV heads in Mistral.
This changes selection semantics relative to legacy per-query-head SnapKV.
"""
import importlib.util
from pathlib import Path
from transformers import DynamicCache
from snapkv_adapter import SnapKVAdapter


def reference_module():
    path=Path(__file__).resolve().parent/'KVCache-Factory/pyramidkv/pyramidkv_utils.py'
    spec=importlib.util.spec_from_file_location('current_gqa_snap_reference',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GQASnapKVAdapter(SnapKVAdapter):
    def __init__(self, model, capacity=1024, window=32, kernel=5, *, reuse_projection=True):
        reference=reference_module()
        super().__init__(model,capacity,window,kernel,reuse_projection=reuse_projection)
        try:
            # The inherited capture/cleanup is proven, but this cache keeps
            # genuine KV heads. Native attention must perform its own GQA.
            for index,groups in self.original_groups.items():
                model.model.layers[index].self_attn.num_key_value_groups=groups
            self.cluster=reference.SnapKVCluster(window,capacity,kernel,'avgpool',
                gqa_score_agg='mean')
        except BaseException:
            self.close()
            raise

    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        if self.get_seq_length(layer_idx)==0:
            query=self.queries.pop(layer_idx)
            compressed=self.cluster.update_kv(key_states,query,value_states,None,
                self.original_groups[layer_idx])
            DynamicCache.update(self,*compressed,layer_idx)
            # Full prefill attention; only subsequent decode sees compression.
            return key_states,value_states
        return DynamicCache.update(self,key_states,value_states,layer_idx)
