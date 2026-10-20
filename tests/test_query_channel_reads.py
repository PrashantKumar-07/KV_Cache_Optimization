"""Exact selected-value delivery and unchanged real-Mistral query behavior."""
import sys
from pathlib import Path
import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from compare_cpu_batched import digest
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy,_attn_impl,to_tuple_kv,apply_tuple_kv_inplace
from query_visible_suffix import SuffixQuerySelector
from query_channel_reads import ChannelQuerySelector,write_selected_values
from test_query_visible_suffix import fixture,rawbits


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
def test_selected_value_suffix_preserves_all_other_bits_and_storage(dtype):
    parents,layer,cold,relative=fixture(dtype)
    before=[t.clone() for t in parents]
    identity=[(id(t),t.data_ptr(),t.storage_offset(),t.stride()) for t in (layer.keys,layer.values)]
    selected=cold[1].index_select(1,relative)
    write_selected_values(layer,7,cold[0],selected,relative)
    assert identity==[(id(t),t.data_ptr(),t.storage_offset(),t.stride()) for t in (layer.keys,layer.values)]
    for parent,previous,source in zip(parents,before,cold):
        want=previous.clone();want[1:2,:,9:12]=source.index_select(1,relative)[None]
        assert torch.equal(rawbits(parent),rawbits(want))


def test_empty_selected_value_suffix_preserves_all_storage():
    parents,layer,cold,relative=fixture();before=[t.clone() for t in parents]
    write_selected_values(layer,10,cold[0],cold[1][:,:0],relative[:0])
    assert all(torch.equal(rawbits(a),rawbits(b)) for a,b in zip(parents,before))


@pytest.mark.parametrize('fault',['key_alias','value_alias','cross_alias','destination_alias',
    'value_quota','dtype','head_stride','row_stride','index_dtype','index_shape','index_scalar',
    'fast_rows','requires_grad','relative_alias'])
def test_invalid_selected_value_geometry_or_alias_rejected_before_either_write(fault):
    parents,layer,cold,relative=fixture();selected=cold[1].index_select(1,relative);fast_rows=7
    if fault=='key_alias': cold[0]=layer.keys[0]
    elif fault=='value_alias': selected=layer.values[0,:,7:]
    elif fault=='cross_alias': selected=layer.keys[0,:,7:]
    elif fault=='destination_alias': layer.values=layer.keys
    elif fault=='value_quota': selected=selected[:,:2]
    elif fault=='dtype': selected=selected.float()
    elif fault=='head_stride': layer.keys=layer.keys.as_strided(layer.keys.shape,(384,0,8,1))
    elif fault=='row_stride': layer.values=layer.values.as_strided(layer.values.shape,(384,128,0,1))
    elif fault=='index_dtype': relative=relative.int()
    elif fault=='index_shape': relative=relative[None]
    elif fault=='index_scalar': relative=torch.tensor(0)
    elif fault=='fast_rows': fast_rows=6
    elif fault=='requires_grad': cold[0]=cold[0].requires_grad_()
    else: relative=layer.keys[0,0,7:9].view(torch.int64).reshape(-1)[:3]
    before=[t.clone() for t in parents]
    with pytest.raises(ValueError): write_selected_values(layer,fast_rows,cold[0],selected,relative)
    assert all(torch.equal(rawbits(a),rawbits(b)) for a,b in zip(parents,before))


@pytest.mark.parametrize('seq,hot,cold', [(8,12,16),(24,8,16),(48,12,24),(64,16,32)])
def test_channel_selector_preserves_real_model_decode(seq,hot,cold):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, sliding_window=None, attn_implementation='eager')).eval()
    rows = []
    for cls in (SuffixQuerySelector,ChannelQuerySelector):
        policy = TieredKVPolicy(sram_budget=hot, stt_budget=cold, start_size=1,
                               recent_size=4, stt_expose_quota=2)
        class CpuReader:
            def indices(self,index): return torch.where(~policy.layer_caches[index].stt_shadow)[0]
            def read_live(self,index):
                return policy.layer_caches[index].select_stt_pair(self.indices(index).tolist())
            def read_keys(self,index):
                key,self.pending_value=self.read_live(index)
                return key
            def read_values(self,index,relative):
                return self.pending_value.index_select(1,relative)
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
                assert selector.visible_suffix_updates==selector.selected_layers
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
