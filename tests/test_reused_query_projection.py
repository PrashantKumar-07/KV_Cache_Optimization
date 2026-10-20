"""Projection reuse must preserve query decisions and release failure context."""
from types import SimpleNamespace
import hashlib,sys
from pathlib import Path
import pytest,torch
from transformers import MistralConfig,MistralForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from pool_query_selector import PoolQuerySelector
from reused_query_projection import ReusedQuerySelector
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy,_attn_impl,to_tuple_kv,apply_tuple_kv_inplace
from compare_cpu_batched import digest


def model():
    torch.manual_seed(0);torch.set_num_threads(2)
    return MistralForCausalLM(MistralConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,sliding_window=None,attn_implementation='eager')).eval()


def prefill(m,policy,seq):
    capture=PromptQueryCapture(m,policy)
    try:
        with _attn_impl(m,'sdpa'):
            out=m((torch.arange(seq)%60+1)[None],use_cache=True,position_ids=torch.arange(seq)[None],logits_to_keep=1)
        past=out.past_key_values;apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)));return past
    finally:capture.close()


@pytest.mark.parametrize('seq,hot,cold',[(8,12,16),(24,8,16),(48,12,24),(64,16,32)])
@torch.inference_mode()
def test_reused_projection_preserves_selector(seq,hot,cold):
    m=model();rows=[]
    for cls in (PoolQuerySelector,ReusedQuerySelector):
        policy=TieredKVPolicy(sram_budget=hot,stt_budget=cold,start_size=1,recent_size=4,stt_expose_quota=2)
        calls=[];observers=[];selector=cls(m,policy)
        for layer in m.model.layers:
            observers.append(layer.self_attn.q_proj.register_forward_hook(
                lambda module,args,out:calls.append(1) if policy.initialized else None))
        try:
            past=prefill(m,policy,seq);frames=[]
            for step in range(6):
                out=m(torch.tensor([[step+1]]),past_key_values=past,use_cache=True,
                    position_ids=torch.tensor([[seq+step]]),output_attentions=True)
                past=out.past_key_values;policy.update_scores(to_tuple_kv(past),out.attentions)
                apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
                frames.append((out.logits.clone(),digest(policy.layer_caches),policy._live_totals(),
                    {i:tuple(v) for i,v in policy._vis_stt_idx.items()}))
            rows.append(frames)
            assert len(calls)==18+(selector.selected_layers if cls is PoolQuerySelector else 0)
            if cls is ReusedQuerySelector:assert not selector.pending and not selector.inside_selection
        finally:
            for observer in observers:observer.remove()
            selector.close();policy.reset()
        assert not selector.handles
    for a,b in zip(*rows):assert torch.equal(a[0],b[0]) and a[1:]==b[1:]


@pytest.mark.parametrize('failure',['before_projection','ranking'])
@torch.inference_mode()
def test_projection_failure_releases_context(monkeypatch,failure):
    m=model();policy=TieredKVPolicy(sram_budget=8,stt_budget=16,start_size=1,recent_size=4,stt_expose_quota=2)
    selector=ReusedQuerySelector(m,policy)
    def fail(*args,**options):raise RuntimeError('injected projection failure')
    try:
        past=prefill(m,policy,24)
        if failure=='before_projection':monkeypatch.setattr(m.model.layers[0].self_attn.q_proj,'forward',fail)
        else:selector.ranking[0]=fail
        with pytest.raises(RuntimeError,match='injected projection failure'):
            m(torch.tensor([[1]]),past_key_values=past,use_cache=True,position_ids=torch.tensor([[24]]),output_attentions=True)
        assert not selector.pending and not selector.inside_selection
    finally:selector.close();policy.reset()
    assert not selector.handles


def test_partial_constructor_removes_installed_hooks():
    m=model();attention=m.model.layers[0].self_attn
    before=tuple(map(len,(attention._forward_pre_hooks,attention._forward_hooks,attention.q_proj._forward_hooks)))
    fake=SimpleNamespace(model=SimpleNamespace(layers=[m.model.layers[0],SimpleNamespace(self_attn=torch.nn.Identity())]))
    with pytest.raises(ValueError,match='native MistralAttention'):ReusedQuerySelector(fake,None)
    assert before==tuple(map(len,(attention._forward_pre_hooks,attention._forward_hooks,attention.q_proj._forward_hooks)))
