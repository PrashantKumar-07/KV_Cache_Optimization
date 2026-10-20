"""Exact cold-suffix mutation, ownership guards and actual Mistral parity."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from compare_cpu_batched import digest
from evaluate_query_selection import PromptQueryCapture
from gpu_query_read_indices import GpuReadQuerySelector
from longbench_eval import TieredKVPolicy, _attn_impl, to_tuple_kv, apply_tuple_kv_inplace
from query_visible_suffix import SuffixQuerySelector, replace_cold_suffix


def rawbits(tensor):
    return tensor.contiguous().view(torch.uint8)


def fixture(dtype=torch.bfloat16):
    parents = [torch.arange(2*3*16*8, dtype=torch.float32).reshape(2,3,16,8).to(dtype) + n
               for n in (0,1000)]
    layer = SimpleNamespace(keys=parents[0][1:2,:,2:12], values=parents[1][1:2,:,2:12])
    cold = [torch.arange(3*11*8, dtype=torch.float32).reshape(3,11,8).to(dtype) + n
            for n in (2000,3000)]
    return parents, layer, cold, torch.tensor([8,0,5], dtype=torch.int64)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_bitwise_suffix_offset_stride_prefix_and_tail(dtype):
    parents, layer, cold, relative = fixture(dtype)
    before = [parent.clone() for parent in parents]
    identities = [id(t) for t in (layer.keys,layer.values)]
    pointers = [t.data_ptr() for t in (layer.keys,layer.values)]
    replace_cold_suffix(layer, 7, *cold, relative)
    assert identities == [id(t) for t in (layer.keys,layer.values)]
    assert pointers == [t.data_ptr() for t in (layer.keys,layer.values)]
    for parent, previous, source in zip(parents,before,cold):
        expected = previous.clone()
        expected[1:2,:,9:12] = source.index_select(1, relative).unsqueeze(0)
        assert torch.equal(rawbits(parent),rawbits(expected))


def test_zero_cold_suffix_leaves_all_storage_untouched():
    parents, layer, cold, relative = fixture()
    before = [parent.clone() for parent in parents]
    replace_cold_suffix(layer, 10, *cold, relative[:0])
    assert all(torch.equal(rawbits(a),rawbits(b)) for a,b in zip(parents,before))


@pytest.mark.parametrize('fault', ['source_alias','cross_source_alias','destination_alias',
    'shape','dtype','cold_shape','head_stride','row_stride','relative_dtype','relative_shape','fast_rows'])
def test_invalid_ownership_and_geometry_rejected_before_either_write(fault):
    parents, layer, cold, relative = fixture()
    fast_rows = 7
    if fault == 'source_alias': cold[0] = layer.keys[0]
    elif fault == 'cross_source_alias': cold[1] = layer.keys[0]
    elif fault == 'destination_alias': layer.values = layer.keys
    elif fault == 'shape': layer.values = layer.values[:,:,:9]
    elif fault == 'dtype': cold[1] = cold[1].float()
    elif fault == 'cold_shape': cold[1] = cold[1][:,:10]
    elif fault == 'head_stride': layer.keys = layer.keys.as_strided(layer.keys.shape, (384,0,8,1))
    elif fault == 'row_stride': layer.values = layer.values.as_strided(layer.values.shape, (384,128,0,1))
    elif fault == 'relative_dtype': relative = relative.int()
    elif fault == 'relative_shape': relative = relative[None]
    else: fast_rows = 6
    before = [parent.clone() for parent in parents]
    with pytest.raises(ValueError): replace_cold_suffix(layer, fast_rows, *cold, relative)
    assert all(torch.equal(rawbits(a),rawbits(b)) for a,b in zip(parents,before))


@pytest.mark.parametrize('seq,hot,cold', [(8,12,16),(24,8,16),(48,12,24),(64,16,32)])
def test_suffix_selector_preserves_real_model_decode(seq,hot,cold):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, sliding_window=None, attn_implementation='eager')).eval()
    rows = []
    for cls in (GpuReadQuerySelector,SuffixQuerySelector):
        policy = TieredKVPolicy(sram_budget=hot, stt_budget=cold, start_size=1,
                               recent_size=4, stt_expose_quota=2)
        class CpuReader:
            def indices(self,index): return torch.where(~policy.layer_caches[index].stt_shadow)[0]
            def read_live(self,index):
                return policy.layer_caches[index].select_stt_pair(self.indices(index).tolist())
        policy.query_reader = CpuReader()
        capture = PromptQueryCapture(model,policy)
        selector = cls(model,policy)
        observations=[];frames=[];handles=[]
        for index, layer in enumerate(model.model.layers):
            def observe(module,args,options,index=index):
                if index not in policy.layer_caches or options['hidden_states'].shape[1]!=1: return
                row=options['past_key_values'].layers[index]
                observations.append((row.keys.clone(),row.values.clone(),tuple(policy._vis_stt_idx[index])))
            handles.append(layer.self_attn.register_forward_pre_hook(observe,with_kwargs=True))
        try:
            with torch.inference_mode():
                ids=(torch.arange(seq)%60+1)[None]
                with _attn_impl(model,'sdpa'):
                    out=model(ids,use_cache=True,position_ids=torch.arange(seq)[None],logits_to_keep=1)
                past=out.past_key_values
                apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
                for step in range(6):
                    out=model(torch.tensor([[step+1]]),past_key_values=past,use_cache=True,
                              position_ids=torch.tensor([[seq+step]]),output_attentions=True)
                    past=out.past_key_values
                    policy.update_scores(to_tuple_kv(past),out.attentions)
                    apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
                    frames.append((out.logits.clone(),digest(policy.layer_caches),policy._live_totals(),
                        {i:tuple(v) for i,v in policy._vis_stt_pos.items()}))
                if cls is SuffixQuerySelector: assert selector.visible_suffix_updates==selector.selected_layers
                rows.append((frames,observations))
        finally:
            selector.close();capture.close()
            for handle in handles: handle.remove()
            policy.reset()
        assert not selector.handles
        assert all(not layer.self_attn._forward_pre_hooks for layer in model.model.layers)
    for a,b in zip(rows[0][0],rows[1][0]):
        assert torch.equal(a[0],b[0]) and a[1:]==b[1:]
    for a,b in zip(rows[0][1],rows[1][1]):
        assert torch.equal(rawbits(a[0]),rawbits(b[0]))
        assert torch.equal(rawbits(a[1]),rawbits(b[1])) and a[2]==b[2]
