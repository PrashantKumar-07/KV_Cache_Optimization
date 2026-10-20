"""Rawbit reservedtail append, conservativefallback and perinstance restoration."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from transformers.cache_utils import DynamicCache,DynamicLayer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from reserved_decode_append import append_reserved,capacity,AppendLease


def fixture():
    roots=[torch.arange(2*8*4,dtype=torch.int16).reshape(1,2,8,4).view(torch.bfloat16).clone() for _ in range(2)]
    layer=DynamicLayer();layer.is_initialized=True;layer.keys,layer.values=(t[:,:,:7] for t in roots)
    new=torch.arange(8,dtype=torch.int16).reshape(1,2,1,4).view(torch.bfloat16).clone()
    return layer,roots,new


def bits(t):return t.contiguous().view(torch.int16)


def test_append_rawbits_prefix_ownedstorage_and_tail_preserved():
    layer,roots,new=fixture();prefix=[t[:,:,:7].clone() for t in roots];ids=[t.untyped_storage()._cdata for t in roots]
    with torch.inference_mode():result=append_reserved(layer,new,new.clone())
    assert result is not None
    for output,root,expected,identity in zip(result,roots,prefix,ids):
        assert output.untyped_storage()._cdata==identity and torch.equal(bits(output[:,:,:7]),bits(expected))
        assert torch.equal(bits(output[:,:,7:]),bits(new)) and capacity(output)==8


@pytest.mark.parametrize('fault',['capacity','grad','dtype','alias','new_alias','offset','batch','multiappend','uninitialized'])
def test_unsupported_append_fallsback_before_any_cache_write(fault):
    layer,roots,new=fixture();value=new.clone()
    if fault=='capacity':layer.keys=layer.keys.clone();layer.values=layer.values.clone()
    elif fault=='grad':new.requires_grad_()
    elif fault=='dtype':new=new.float();value=value.float()
    elif fault=='alias':layer.values=layer.keys
    elif fault=='new_alias':new=roots[0][:,:,:1]
    elif fault=='offset':layer.keys=roots[0][:,:,1:]
    elif fault=='batch':new=new.expand(2,-1,-1,-1);value=value.expand(2,-1,-1,-1)
    elif fault=='multiappend':new=new.expand(-1,-1,2,-1);value=value.expand(-1,-1,2,-1)
    else:layer.is_initialized=False
    before=[r.clone() for r in roots]
    with torch.inference_mode():assert append_reserved(layer,new,value) is None
    assert all(torch.equal(bits(a),bits(b)) for a,b in zip(roots,before))


def test_training_mode_preserves_original_update_path():
    layer,roots,new=fixture();assert append_reserved(layer,new,new.clone()) is None


class Model(torch.nn.Module):
    config=SimpleNamespace(model_type='mistral')
    def __init__(self,fail=False):super().__init__();self.fail=fail
    def forward(self,ids,*,past_key_values,use_cache=True):
        new=torch.zeros(1,2,1,4,dtype=torch.bfloat16)
        for layer in past_key_values.layers:
            layer.update(new,new.clone())
            if self.fail:raise ValueError('partial forward')
        return past_key_values


@pytest.mark.parametrize('fail',[False,True])
def test_forward_restores_class_methods_and_hooks_on_partial_failure(fail):
    policy=type('Policy',(),{'initialized':True})();model=Model(fail);past=DynamicCache()
    layer,roots,new=fixture();past.layers=[layer]
    lease=AppendLease(model,policy)
    with torch.inference_mode():
        if fail:
            with pytest.raises(ValueError,match='partial'):model(torch.ones(1,1,dtype=torch.long),past_key_values=past,use_cache=True)
        else:model(torch.ones(1,1,dtype=torch.long),past_key_values=past,use_cache=True)
    assert not lease.pending and 'update' not in layer.__dict__ and lease.fast_updates==1
    lease.close();assert not model._forward_hooks and not model._forward_pre_hooks


@pytest.mark.parametrize('seq',[8,24,48])
def test_real_tiny_mistral_reserved_append_preserves_logits_and_cache(seq):
    from transformers import MistralForCausalLM,MistralConfig
    from longbench_eval import TieredKVPolicy,_attn_impl
    from longbench_eval import to_tuple_kv,apply_tuple_kv_inplace
    from compare_cpu_batched import digest
    from reserved_decode_append import reserve_visible
    torch.set_num_threads(2);torch.manual_seed(0)
    model=MistralForCausalLM(MistralConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,vocab_size=64,sliding_window=None,attn_implementation='eager')).to(dtype=torch.bfloat16).eval()
    class Candidate(TieredKVPolicy):
        def __init__(self,**options):
            self.reserved_buffer_allocations=self.shared_hot_prefixes=0
            super().__init__(**options)
        def _visible_tuple(self,past):return reserve_visible(self,super()._visible_tuple(past))
    rows=[]
    for cls in [TieredKVPolicy,Candidate]:
        policy=cls(sram_budget=12,stt_budget=16,start_size=1,recent_size=4,stt_expose_quota=2,kv_dtype=torch.bfloat16)
        lease=AppendLease(model,policy) if cls is Candidate else None
        frames=[]
        try:
            with torch.inference_mode():
                ids=(torch.arange(seq)%60+1)[None]
                with _attn_impl(model,'sdpa'):out=model(ids,use_cache=True,position_ids=torch.arange(seq)[None],logits_to_keep=1)
                past=out.past_key_values;apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
                for step in range(6):
                    out=model(torch.tensor([[step+1]]),past_key_values=past,use_cache=True,position_ids=torch.tensor([[seq+step]]),output_attentions=True)
                    past=out.past_key_values;policy.update_scores(to_tuple_kv(past),out.attentions)
                    apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
                    frames.append((out.logits.clone(),digest(policy.layer_caches),policy._live_totals(),{i:tuple(v) for i,v in policy._vis_stt_idx.items()}))
            if lease:assert lease.fast_updates>0 and not lease.pending
            rows.append(frames)
        finally:
            if lease:lease.close()
            policy.reset()
        assert not model._forward_hooks and not model._forward_pre_hooks
    for a,b in zip(*rows):assert torch.equal(a[0],b[0]) and a[1:]==b[1:]
