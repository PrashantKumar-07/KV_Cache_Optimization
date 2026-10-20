"""Tiny-model integration gate; run directly on CPU."""
import sys
import torch
from transformers import MistralConfig, MistralForCausalLM
from snapkv_adapter import SnapKVAdapter


def main():
    torch.manual_seed(7)
    config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, sliding_window=None)
    config._attn_implementation = sys.argv[1] if len(sys.argv) > 1 else 'eager'
    model = MistralForCausalLM(config).eval()
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    dtype = torch.bfloat16 if len(sys.argv) > 3 else torch.float32
    model = model.to(dtype)
    ids = torch.randint(0, 64, (1, length))
    with torch.no_grad():
        full = model(ids, use_cache=True)
        cache = SnapKVAdapter(model, capacity=12, window=4)
        expected = []
        select = cache.cluster.update_kv
        def record(k, q, v, mask, groups):
            result = select(k, q, v, mask, groups)
            expected.append(tuple(x.clone() for x in result))
            return result
        cache.cluster.update_kv = record
        try:
            out = model(ids, past_key_values=cache, use_cache=True,
                        position_ids=torch.arange(length).unsqueeze(0))
            torch.testing.assert_close(out.logits, full.logits, atol=2e-3 if dtype == torch.bfloat16 else 1e-6, rtol=1e-5)
            assert len(expected) == 2
            for layer, (k, v) in zip(cache.layers, expected):
                torch.testing.assert_close(layer.keys, k)
                torch.testing.assert_close(layer.values, v)
                assert k.shape == (1, 4, min(length, 12), 8)
            # Explicit positions preserve original sequence coordinates after compression.
            for pos in range(length, length + 3):
                token = out.logits[:, -1:].argmax(-1)
                out = model(token, past_key_values=cache, use_cache=True,
                            position_ids=torch.tensor([[pos]]))
                assert torch.isfinite(out.logits).all()
                assert all(layer.keys.shape[-2] == min(length, 12) + pos - length + 1 for layer in cache.layers)
                if length < 12:
                    # With no prompt eviction, every decode logit must agree with
                    # ordinary GQA, independently checking expanded cache semantics.
                    for layer in model.model.layers:
                        layer.self_attn.num_key_value_groups = 2
                    full = model(token, past_key_values=full.past_key_values,
                                 use_cache=True, position_ids=torch.tensor([[pos]]))
                    for layer in model.model.layers:
                        layer.self_attn.num_key_value_groups = 1
                    torch.testing.assert_close(out.logits, full.logits,
                        atol=2e-3 if dtype == torch.bfloat16 else 1e-6, rtol=1e-5)
            assert not cache.queries
        finally:
            cache.close()
        assert all(layer.self_attn.num_key_value_groups == 2 for layer in model.model.layers)
    print('PASS: full-prefill logits, reference-selected cache, expanded heads, growing decode, hook cleanup')


if __name__ == '__main__':
    main()
