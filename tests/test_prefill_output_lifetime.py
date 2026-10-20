"""Prefill outputs must not survive into the first decode allocation."""
import sys
import weakref
from pathlib import Path

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from longbench_eval import TieredKVPolicy, generate_with_policy, prefill_logit_kwargs


class Tokenizer:
    eos_token_id = -1

    def __call__(self, text, **options):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]]))

    def decode(self, ids, **options):
        return ' '.join(map(str, ids))


@pytest.mark.parametrize('chunk,max_gen,eos', [(0, 3, False), (4, 3, False),
    (0, 1, False), (4, 1, False), (0, 3, True), (4, 3, True), (0, 0, False)])
@pytest.mark.parametrize('last_logits', [False, True])
def test_prefill_output_expires_before_decode(chunk, max_gen, eos, last_logits):
    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=32, sliding_window=None, attn_implementation='eager')).eval()
    tokenizer = Tokenizer()
    policy = TieredKVPolicy(sram_budget=16, stt_budget=16, start_size=1,
        recent_size=2, prefill_chunk_size=chunk)
    refs = {}; calls = []

    def before(module, positional, options):
        calls.append(bool(options.get('output_attentions')))
        if options.get('output_attentions'):
            assert refs['output']() is None, 'Unused prefill output retained during decode'
            assert refs['logits']() is None, 'Unused prefill logits retained during decode'

    def after(module, positional, options, out):
        if not options.get('output_attentions'):
            expected_rows = 1 if last_logits else positional[0].shape[1]
            assert out.logits.shape == (1, expected_rows, 32)
            refs.update(output=weakref.ref(out), logits=weakref.ref(out.logits))
            if eos:
                tokenizer.eos_token_id = out.logits[:, -1:].argmax(-1).item()

    handles = [model.register_forward_pre_hook(before, with_kwargs=True),
        model.register_forward_hook(after, with_kwargs=True)]
    try:
        text = generate_with_policy(model, tokenizer, 'prompt', max_gen, policy, 'cpu',
            prefill_last_logits=last_logits)
    finally:
        for handle in handles:
            handle.remove()
    if max_gen == 0:
        assert text == '' and calls == []
    else:
        assert len(text.split()) == (1 if eos else max_gen)
        assert calls.count(True) == (0 if eos else max_gen - 1)
        assert policy.last_generation == {'prompt_tokens': 8,
            'generated_tokens': 1 if eos else max_gen}
        assert refs['output']() is None and refs['logits']() is None


def test_last_logits_requires_explicit_native_support():
    class Unsupported(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            raise AssertionError('Unsupported model must not be called')

    assert prefill_logit_kwargs(Unsupported()) == {}
    with pytest.raises(ValueError, match='requires native'):
        generate_with_policy(Unsupported(), Tokenizer(), 'prompt', 3,
            TieredKVPolicy(), 'cpu', prefill_last_logits=True)


def test_latency_profiler_last_logits_preserves_fixed_stream():
    from latency_profile import profile_one
    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=32, sliding_window=None, attn_implementation='eager')).eval()
    predictions = []
    for last in (False, True):
        shapes = []
        def after(module, positional, options, out):
            shapes.append(list(out.logits.shape))
        handle = model.register_forward_hook(after, with_kwargs=True)
        try:
            result = profile_one(model, Tokenizer(), TieredKVPolicy(
                sram_budget=16, stt_budget=16, start_size=1, recent_size=2), 8,
                n_decode=3, n_warmup=0, n_repeat=1, device='cpu',
                forced_token_ids=[1, 2, 3, 4], prefill_last_logits=last)
        finally:
            handle.remove()
        assert shapes == [[1, 1 if last else 8, 32], *([[1, 1, 32]] * 3)]
        predictions.append(result['trials'][0]['predicted_token_ids'])
    assert predictions[0] == predictions[1]
