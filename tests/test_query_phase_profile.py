"""Profiler annotations preserve logits, cache state and visible selections."""
import sys
from pathlib import Path
import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from compare_cpu_batched import digest
from evaluate_query_selection import PromptQueryCapture
from gpu_query_read_indices import GpuReadQuerySelector
from longbench_eval import TieredKVPolicy, _attn_impl, to_tuple_kv, apply_tuple_kv_inplace
from profile_query_phases import ProfiledGpuQuerySelector


@pytest.mark.parametrize('seq,hot,cold', [(8,12,16), (24,8,16), (48,12,24), (64,16,32)])
def test_profiler_annotations_preserve_decode(seq, hot, cold):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, sliding_window=None, attn_implementation='eager')).eval()
    rows = []
    for cls in (GpuReadQuerySelector, ProfiledGpuQuerySelector):
        policy = TieredKVPolicy(sram_budget=hot, stt_budget=cold, start_size=1,
                               recent_size=4, stt_expose_quota=2)

        class CpuReader:
            def indices(self, index):
                return torch.where(~policy.layer_caches[index].stt_shadow)[0]

            def read_live(self, index):
                return policy.layer_caches[index].select_stt_pair(self.indices(index).tolist())

        policy.query_reader = CpuReader()
        capture = PromptQueryCapture(model, policy)
        selector = cls(model, policy)
        try:
            with torch.inference_mode():
                ids = (torch.arange(seq) % 60 + 1)[None]
                with _attn_impl(model, 'sdpa'):
                    out = model(ids, use_cache=True, position_ids=torch.arange(seq)[None], logits_to_keep=1)
                past = out.past_key_values
                apply_tuple_kv_inplace(past, policy(to_tuple_kv(past)))
                frames = []
                for step in range(6):
                    out = model(torch.tensor([[step+1]]), past_key_values=past, use_cache=True,
                        position_ids=torch.tensor([[seq+step]]), output_attentions=True)
                    past = out.past_key_values
                    policy.update_scores(to_tuple_kv(past), out.attentions)
                    apply_tuple_kv_inplace(past, policy(to_tuple_kv(past)))
                    frames.append((out.logits.clone(), digest(policy.layer_caches),
                        policy._live_totals(), {i:tuple(v) for i,v in policy._vis_stt_pos.items()}))
                rows.append(frames)
        finally:
            selector.close()
            capture.close()
            policy.reset()
        assert not selector.handles
        assert all(not layer.self_attn._forward_pre_hooks for layer in model.model.layers)
    for original, instrumented in zip(*rows):
        assert torch.equal(original[0], instrumented[0])
        assert original[1:] == instrumented[1:]
