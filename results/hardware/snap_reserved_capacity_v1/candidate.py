"""Research faithful SnapKV selector with bounded genuine future KV backing."""
import torch
from snapkv_adapter import SnapKVAdapter


class ReservedSnapKV(SnapKVAdapter):
    def __init__(self,model,capacity=974,window=32,kernel=5,*,max_rows=1024):
        if type(max_rows) is not int or max_rows < capacity:
            raise ValueError('Reserved rows must cover prompt capacity')
        self.max_rows=max_rows
        self.roots={}
        super().__init__(model,capacity,window,kernel)

    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        if self.get_seq_length(layer_idx)==0:
            result=super().update(key_states,value_states,layer_idx,*args,**kwargs)
            layer=self.layers[layer_idx]
            key,value=layer.keys,layer.values
            if key.shape!=value.shape or key.shape[0]!=1 or key.shape[-2]>self.max_rows:
                raise ValueError('Require batch-one full prefill within reserved capacity')
            shape=(*key.shape[:-2],self.max_rows,key.shape[-1])
            roots=tuple(torch.empty(shape,dtype=t.dtype,device=t.device) for t in (key,value))
            self.roots[layer_idx]=roots
            length=key.shape[-2]
            for root,data in zip(roots,(key,value)):root.narrow(-2,0,length).copy_(data)
            layer.keys,layer.values=(root.narrow(-2,0,length) for root in roots)
            return result  # Full native expanded prompt KV, unchanged prefill attention.
        layer=self.layers[layer_idx];length=layer.keys.shape[-2]
        if (key_states.shape!=value_states.shape or key_states.ndim!=4
                or key_states.shape[0]!=1 or key_states.shape[-2]!=1
                or length+1>self.max_rows):
            raise ValueError('Only one-token decode within reserved capacity is supported')
        groups=self.original_groups[layer_idx]
        key=self.reference.repeat_kv(key_states,groups)
        value=self.reference.repeat_kv(value_states,groups)
        roots=self.roots[layer_idx]
        if any(data.dtype!=root.dtype or data.device!=root.device
               or data.shape[1]!=root.shape[1] or data.shape[-1]!=root.shape[-1]
               for root,data in zip(roots,(key,value))):
            raise ValueError('Decode geometry/dtype/device differs from reserved backing')
        for root,data in zip(roots,(key,value)):root.narrow(-2,length,1).copy_(data)
        layer.keys,layer.values=(root.narrow(-2,0,length+1) for root in roots)
        return layer.keys,layer.values

    def close(self):
        super().close()
        self.roots.clear()
