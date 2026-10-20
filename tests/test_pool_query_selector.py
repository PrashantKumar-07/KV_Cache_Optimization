"""Paired-storage selector must preserve existing GPU-selector math on CPU."""
import sys
from pathlib import Path
import pytest
import torch
from transformers import MistralConfig,MistralForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from pool_query_selector import PoolQuerySelector
from evaluate_query_selection import QuerySelector,PromptQueryCapture
from longbench_eval import TieredKVPolicy,_attn_impl,to_tuple_kv,apply_tuple_kv_inplace
from compare_cpu_batched import digest


@pytest.mark.parametrize('seq,hot,cold',[(8,12,16),(24,8,16),(48,12,24),(64,16,32)])
def test_paired_selector_matches_existing_selector(seq,hot,cold):
    torch.set_num_threads(2);torch.manual_seed(0)
    model=MistralForCausalLM(MistralConfig(hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,vocab_size=64,
        sliding_window=None,attn_implementation='eager')).eval()
    rows=[]
    for cls in (QuerySelector,PoolQuerySelector):
        policy=TieredKVPolicy(sram_budget=hot,stt_budget=cold,start_size=1,recent_size=4,stt_expose_quota=2)
        capture=PromptQueryCapture(model,policy);selector=cls(model,policy)
        try:
            with torch.inference_mode():
                ids=(torch.arange(seq)%60+1)[None]
                with _attn_impl(model,'sdpa'):
                    out=model(ids,use_cache=True,position_ids=torch.arange(seq)[None],logits_to_keep=1)
                past=out.past_key_values;apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
                frames=[]
                for step in range(6):
                    out=model(torch.tensor([[step+1]]),past_key_values=past,use_cache=True,
                        position_ids=torch.tensor([[seq+step]]),output_attentions=True)
                    past=out.past_key_values;policy.update_scores(to_tuple_kv(past),out.attentions)
                    apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
                    frames.append((out.logits.clone(),digest(policy.layer_caches),policy._live_totals(),
                        {i:tuple(v) for i,v in policy._vis_stt_pos.items()}))
                rows.append(frames)
        finally:
            selector.close();capture.close();policy.reset()
        assert not selector.handles if cls is PoolQuerySelector else True
    for a,b in zip(*rows):
        assert torch.equal(a[0],b[0])
        assert a[1:]==b[1:]
