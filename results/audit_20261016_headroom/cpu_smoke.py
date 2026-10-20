"""Tiny-model execution proof for wrapper reuse; not thesis evaluation results."""
import importlib.util
import json
from pathlib import Path
import torch
from transformers import MistralConfig, MistralForCausalLM

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('headroom_runner', BASE / 'run_headroom.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    torch.manual_seed(0)
    config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, sliding_window=None)
    config._attn_implementation = 'eager'
    model = MistralForCausalLM(config).to(torch.bfloat16).eval()
    class Tokenizer:
        eos_token_id = None
        def decode(self, ids, skip_special_tokens=True):
            return 'hello'
    ids = torch.randint(0, 64, (1, 2300))
    runner.bench.DATASET_TO_MAXGEN['qasper'] = 3
    original_constructor = runner.frozen.policy_for
    results = {}
    with torch.inference_mode():
        model(torch.tensor([[1]]), use_cache=False, output_attentions=True, logits_to_keep=1)
        baseline = runner.frozen.run_request(model, Tokenizer(), ids, 'qasper',
            {'answers': ['hello']}, 'tiered_hot1024')
        for fraction in [1., .75, .5, .25]:
            result = runner.run_request(model, Tokenizer(), ids, 'qasper',
                {'answers': ['hello']}, fraction)
            assert result['generated_tokens'] == 3
            assert result['generated_token_ids'][0] == baseline['generated_token_ids'][0]
            assert result['all_model_logits_finite'] and result['hooks_and_groups_restored']
            assert runner.frozen.policy_for is original_constructor
            if fraction == .75:
                for key in ['prediction', 'generated_token_ids', 'cache_stats',
                            'geometry_after_prefill_before_first_decode',
                            'geometry_after_last_model_call_and_policy_commit',
                            'decode_attended_width']:
                    assert result[key] == baseline[key], f'Wrapper.75 parity failed: {key}'
            results[str(fraction)] = {'generated_token_ids': result['generated_token_ids'],
                'cache_stats': result['cache_stats'],
                'decode_attended_width': result['decode_attended_width'],
                'initial_layer0_visible_shape': result['geometry_after_prefill_before_first_decode']['layers'][0]['visible_k']['shape'],
                'initial_layer0_cold_live_rows': result['geometry_after_prefill_before_first_decode']['layers'][0]['cold_live_rows'],
                'initial_layer0_cold_shadow_rows': result['geometry_after_prefill_before_first_decode']['layers'][0]['cold_shadow_rows']}
    report = {'passed': True, 'scope': 'Actual tiny CPU random-model wrapper smoke, not thesis quality results',
        'model': '2-layer BF16 Mistral, Qheads4/KVheads2, hidden32, vocab64',
        'prompt_tokens': 2300, 'max_gen': 3, 'all_fractions_common_first_token': True,
        'fraction075_exact_original_generator_parity': True,
        'constructor_restored_after_each_fraction': True, 'results': results}
    path = BASE / 'cpu_smoke.json'
    assert not path.exists(), 'Preserve prior evidence'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
