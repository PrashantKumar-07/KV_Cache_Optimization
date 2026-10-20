"""Fixed-token profiling works for one-tier and tiered counter schemas."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import latency_profile as profile
from longbench_eval import FullCachePolicy, StreamingLLMPolicy, H2OPolicy, SnapKVPolicy, TieredKVPolicy
from test_prefill_output_lifetime import Tokenizer


def policy_for(name):
    return {'full': lambda: FullCachePolicy(),
        'streaming': lambda: StreamingLLMPolicy(1, 5),
        'h2o': lambda: H2OPolicy(6, 1, 2),
        'snapkv': lambda: SnapKVPolicy(6, 1, 2),
        'tiered': lambda: TieredKVPolicy(sram_budget=6, stt_budget=8,
            start_size=1, recent_size=2)}[name]()


@pytest.mark.parametrize('name', ['full', 'streaming', 'h2o', 'snapkv', 'tiered'])
@pytest.mark.parametrize('last_logits', [False, True])
def test_fixed_stream_counter_scope_and_warmup(name, last_logits):
    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=32, sliding_window=None, attn_implementation='eager')).eval()
    calls = []

    def before(module, positional, options):
        calls.append((positional[0].tolist(), options['position_ids'].tolist()))

    handle = model.register_forward_pre_hook(before, with_kwargs=True)
    policy = policy_for(name)
    try:
        result = profile.profile_one(model, Tokenizer(), policy, 8, n_decode=3,
            n_warmup=1, n_repeat=2, device='cpu', forced_token_ids=[1, 2, 3, 4],
            prefill_last_logits=last_logits)
    finally:
        handle.remove()
    assert len(calls) == 12 and len(result['trials']) == 2
    for start in (0, 4, 8):
        assert calls[start] == ([[3, 4, 5, 6, 7, 8, 9, 10]], [list(range(8))])
        assert calls[start + 1:start + 4] == [([[token]], [[8 + step]])
            for step, token in enumerate([1, 2, 3])]
    for trial in result['trials']:
        assert trial['decode_steps'] == 3
        assert len(trial['step_ms']) == len(trial['predicted_token_ids']) == 3
        if name == 'tiered':
            for key in ('paid_writes', 'writes_saved', 'demoted', 'promoted'):
                assert key in trial and trial[key] >= 0
        else:
            assert all(key not in trial for key in ('paid_writes', 'writes_saved', 'demoted', 'promoted'))
    assert result['trials'][0]['predicted_token_ids'] == result['trials'][1]['predicted_token_ids']
