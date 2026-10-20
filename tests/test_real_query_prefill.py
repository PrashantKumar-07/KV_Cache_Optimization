"""Focused gate for the opt-in actual-query prompt selector."""
import sys
from pathlib import Path

import torch
import pytest
from transformers import MistralConfig, MistralForCausalLM

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src', 'experiments')]
from attention import gqa_causal_importance, snapkv_importance
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy, to_tuple_kv


def test_causal_window_excludes_future_keys():
    q = torch.tensor([[[1.0], [0.0]]])  # positions 2 and 3 of four
    k = torch.tensor([[[0.0], [0.0], [0.0], [100.0]]])
    old = snapkv_importance(q, k, pool_kernel=1)
    causal = snapkv_importance(q, k, pool_kernel=1, causal=True)
    assert old[3] > 1.2
    assert causal[3] < 0.3
    torch.testing.assert_close(causal.sum(), torch.tensor(2.0))


@pytest.mark.parametrize('tokens,width', [(1, 1), (13, 5), (32, 8)])
def test_chunked_gqa_scores_match_expanded_head_scores(tokens, width):
    torch.manual_seed(tokens)
    q = torch.randn(4, width, 8)
    k = torch.randn(2, tokens, 8)
    expected = snapkv_importance(q, k.repeat_interleave(2, dim=0),
                                 pool_kernel=5, causal=True)
    actual = gqa_causal_importance(q, k, pool_kernel=5, query_chunk=2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual.argsort(descending=True), expected.argsort(descending=True))


@pytest.mark.parametrize('length', [1, 12, 13, 24, 32])
def test_real_query_capture_consumed_without_changing_prefill_logits(length):
    torch.manual_seed(11)
    model = MistralForCausalLM(MistralConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, sliding_window=None)).eval()
    ids = torch.randint(0, 64, (1, length))
    policy = TieredKVPolicy(sram_budget=12, stt_budget=16, start_size=1,
                            recent_size=4, kv_dtype=torch.float32)
    with torch.no_grad():
        control = model(ids, use_cache=True)
        capture = PromptQueryCapture(model, policy)
        try:
            candidate = model(ids, use_cache=True)
        finally:
            capture.close()
        torch.testing.assert_close(candidate.logits, control.logits)
        assert set(policy.prompt_q_by_layer) == {0, 1}
        assert all(q.shape == (4, min(4, length), 8)
                   for q in policy.prompt_q_by_layer.values())
        original = to_tuple_kv(candidate.past_key_values)
        policy(original)
        assert not policy.prompt_q_by_layer
        assert all(len(c.sram_pos) <= 12 and len(c.stt_pos) <= 16
                   for c in policy.layer_caches.values())
        for i, cache in policy.layer_caches.items():
            keys, values = original[i]
            torch.testing.assert_close(cache.sram_k, keys[0, :, cache.sram_pos])
            torch.testing.assert_close(cache.sram_v, values[0, :, cache.sram_pos])
            torch.testing.assert_close(cache.stt_k, keys[0, :, cache.stt_pos])
            torch.testing.assert_close(cache.stt_v, values[0, :, cache.stt_pos])
