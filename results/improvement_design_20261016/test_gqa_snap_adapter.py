import math
import torch
import pytest
from transformers import DynamicCache, MistralConfig, MistralForCausalLM
from gqa_snap_adapter import GQASnapKVAdapter, reference_module


def independent_select(key,query,value,capacity,window,kernel):
    batch,heads,length,dim=key.shape
    groups=query.shape[1]//heads
    expanded=key.repeat_interleave(groups,dim=1)
    logits=query[:,:,-window:]@expanded.transpose(-1,-2)/math.sqrt(dim)
    positions=torch.arange(length-window,length)
    mask=torch.arange(length)[None,:]>positions[:,None]
    logits=logits.masked_fill(mask[None,None],float('-inf'))
    probability=torch.softmax(logits.float(),dim=-1).to(query.dtype)
    scores=probability[:,:,:,:-window].sum(-2)
    scores=scores.reshape(batch,heads,groups,length-window).mean(2)
    pooled=torch.nn.functional.avg_pool1d(scores,kernel,stride=1,padding=kernel//2)
    ids=pooled.topk(capacity-window,dim=-1).indices
    ids=ids[:,:,:,None].expand(-1,-1,-1,dim)
    return tuple(torch.cat((tensor[:,:,:-window].gather(2,ids),tensor[:,:,-window:]),2) for tensor in (key,value))


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('groups',[1,2,4])
def test_upstream_selector_matches_independent_group_mean(dtype,groups):
    torch.manual_seed(29)
    key=torch.randn(1,2,24,4).to(dtype)
    value=torch.arange(24)[None,None,:,None].expand_as(key).to(dtype)
    query=torch.randn(1,2*groups,24,4).to(dtype)
    cluster=reference_module().SnapKVCluster(4,12,5,'avgpool',gqa_score_agg='mean')
    actual=cluster.update_kv(key,query,value,None,groups)
    expected=independent_select(key,query,value,12,4,5)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert actual[0].shape==(1,2,12,4)


class IndependentAdapter(GQASnapKVAdapter):
    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        if self.get_seq_length(layer_idx)==0:
            query=self.queries.pop(layer_idx)
            compressed=(key_states,value_states) if query.shape[2]<12 else independent_select(
                key_states,query,value_states,12,4,5)
            DynamicCache.update(self,*compressed,layer_idx)
            return key_states,value_states
        return DynamicCache.update(self,key_states,value_states,layer_idx)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('implementation',['eager','sdpa'])
@pytest.mark.parametrize('length',[8,24])
def test_real_model_bridge_matches_independent_cache_and_full_prefill(dtype,implementation,length):
    torch.manual_seed(31)
    config=MistralConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=128,sliding_window=None)
    config._attn_implementation=implementation
    model=MistralForCausalLM(config).eval().to(dtype)
    ids=torch.randint(0,64,(1,length))
    with torch.inference_mode():
        # Initialize framework hooks before snapshots if the environment adds any.
        full_logits=model(ids,use_cache=True).logits
        before=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
        records=[]
        for cls in (GQASnapKVAdapter,IndependentAdapter):
            cache=cls(model,12,4,5)
            states=[]
            try:
                for step in range(4):
                    tokens=ids if step==0 else torch.tensor([[step+1]])
                    positions=torch.arange(length)[None] if step==0 else torch.tensor([[length+step-1]])
                    out=model(tokens,past_key_values=cache,use_cache=True,position_ids=positions)
                    if step==0:
                        assert torch.equal(out.logits,full_logits)
                    states.append((out.logits.clone(),[(layer.keys.clone(),layer.values.clone()) for layer in cache.layers]))
                    assert all(layer.keys.shape[1]==2 for layer in cache.layers)
                    assert not cache.queries and not cache.pending_positions
                records.append(states)
            finally:
                cache.close();cache.close()
            assert all(layer.self_attn.num_key_value_groups==2 for layer in model.model.layers)
            assert before==[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
        for left,right in zip(*records):
            assert torch.equal(left[0],right[0])
            assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(left[1],right[1]))
