"""Actual projection reuse must preserve reference adapter logits and KV bytes."""
from pathlib import Path
import sys
import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from snapkv_adapter import SnapKVAdapter

@pytest.mark.parametrize('implementation',['eager','sdpa'])
@pytest.mark.parametrize('length',[8,24])
def test_reused_projection_matches_reference_with_one_projection(implementation,length):
    torch.manual_seed(13)
    config=MistralConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=128,sliding_window=None)
    config._attn_implementation=implementation
    model=MistralForCausalLM(config).eval().to(torch.bfloat16)
    ids=torch.randint(0,64,(1,length));stream=[3,7,11];records=[]
    for reuse in [False,True]:
        counts={'q':0,'k':0};handles=[]
        def count(name):
            def hook(*args):counts[name]+=1
            return hook
        for layer in model.model.layers:
            handles.append(layer.self_attn.q_proj.register_forward_hook(count('q')))
            handles.append(layer.self_attn.k_proj.register_forward_hook(count('k')))
        cache=SnapKVAdapter(model,12,4,reuse_projection=reuse)
        try:
            with torch.inference_mode():
                logits=[];states=[]
                for step in range(4):
                    tokens=ids if step==0 else torch.tensor([[stream[step-1]]])
                    positions=torch.arange(length).unsqueeze(0) if step==0 else torch.tensor([[length+step-1]])
                    out=model(tokens,past_key_values=cache,use_cache=True,position_ids=positions)
                    logits.append(out.logits.clone())
                    states.append([(l.keys.clone(),l.values.clone()) for l in cache.layers])
                    assert not cache.pending_positions and not cache.queries
                records.append((logits,states))
            assert counts==({'q':8,'k':8} if reuse else {'q':10,'k':10})
        finally:
            cache.close()
            for handle in handles:handle.remove()
        assert all(l.self_attn.num_key_value_groups==2 for l in model.model.layers)
    for left,right in zip(records[0][0],records[1][0]):
        assert torch.equal(left,right)
    for left,right in zip(records[0][1],records[1][1]):
        assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(left,right))

def test_projection_registration_failure_restores_attention(monkeypatch):
    config=MistralConfig(vocab_size=16,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
    model=MistralForCausalLM(config);a=model.model.layers[0].self_attn
    existing=a.register_forward_pre_hook(lambda *args:None)
    before=dict(a._forward_pre_hooks)
    def fail(*args,**kwargs):raise RuntimeError('projection registration failed')
    monkeypatch.setattr(a.q_proj,'register_forward_hook',fail)
    with pytest.raises(RuntimeError,match='projection registration failed'):
        SnapKVAdapter(model,12,4,reuse_projection=True)
    assert dict(a._forward_pre_hooks)==before and a.num_key_value_groups==2
    existing.remove()
