"""Future-cache ownership and global cold-tail handoff boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from early_shared_cpu_phase import EarlySharedCache,SharedPrefillLease
from reserved_decode_append import capacity
from reuse_visible_policy import ReuseVisiblePolicy

class Policy:
    def __init__(self):
        self.initialized=False;self.stt_expose_quota=2
        self.shared_prefill_geometry=None;self.shared_prefill_layers=0
        self.placed_layers=[];self.resets=0
        self.cache=SimpleNamespace(sram_pos=list(range(4)),stt_pos=[8,9],
            sram_k=torch.arange(16).reshape(2,4,2).to(torch.bfloat16),
            sram_v=-torch.arange(16).reshape(2,4,2).to(torch.bfloat16))
    def place(self,layer,key,value):return self.cache
    def reset(self):self.resets+=1;self.shared_prefill_geometry=None

def test_current_attention_keeps_original_KV_while_future_hot_shares_owner():
    p=Policy();c=EarlySharedCache(p)
    key=torch.arange(40).reshape(1,2,10,2).to(torch.bfloat16);value=-key
    before=[t.clone() for t in (key,value)];hot=p.cache.sram_k.clone()
    result=c.update(key,value,0)
    assert result[0] is key and result[1] is value
    assert torch.equal(key,before[0]) and torch.equal(value,before[1])
    raw=c.layers[0].keys
    assert raw.shape==(1,2,6,2) and capacity(raw)==7
    assert p.cache.sram_k.data_ptr()==raw.data_ptr() and p.cache.sram_k.stride()==raw.stride()[1:]
    assert torch.equal(p.cache.sram_k,hot) and p.shared_prefill_layers==1

def test_global_handoff_fills_cold_tail_without_reallocating_visible_root():
    p=Policy();c=EarlySharedCache(p);key=torch.zeros(1,2,10,2,dtype=torch.bfloat16)
    c.update(key,key.clone(),0)
    for raw,hot in [(c.layers[0].keys,p.cache.sram_k),(c.layers[0].values,p.cache.sram_v)]:
        cold=torch.arange(8).reshape(2,2,2).to(torch.bfloat16)+30
        expected=torch.cat((hot.clone(),cold),dim=1)
        root=raw.untyped_storage()._cdata
        visible=ReuseVisiblePolicy._compose_visible(hot,cold,raw)
        assert torch.equal(visible,expected)
        assert visible.untyped_storage()._cdata==root and capacity(raw)==7

def test_unsupported_dtype_rejects_before_placement_and_partial_lease_resets():
    p=Policy();c=EarlySharedCache(p)
    with pytest.raises(ValueError,match='BF16'):c.update(torch.zeros(1,2,10,2),torch.zeros(1,2,10,2),0)
    assert p.shared_prefill_layers==0 and not c.layers
    model=torch.nn.Module();model.model=torch.nn.Module();model.model.layers=torch.nn.ModuleList([torch.nn.Identity()])
    before=len(model._forward_pre_hooks),len(model._forward_hooks)
    lease=SharedPrefillLease(model,p)
    try:
        args,opts=lease.before(model,(torch.ones(1,10,dtype=torch.long),),{'use_cache':True})
        assert isinstance(opts['past_key_values'],EarlySharedCache)
        lease.after(model,args,opts,None)
        assert p.resets==1
    finally:lease.close()
    assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
