"""Real-model exact selection and prediction gate for boolean live-row gather."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy, generate_with_policy
from provenance import atomic_json, runtime_identity


class RecordingPolicy(TieredKVPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []

    def _visible_tuple(self, past_kv):
        result = super()._visible_tuple(past_kv)
        self.history.append(tuple(tuple(self._vis_stt_idx.get(i, ()))
                                  for i in sorted(self.layer_caches)))
        return result


def run(model, tokenizer, prompt, ctx, use_compaction=False):
    policy = RecordingPolicy(sram_budget=992, stt_budget=3104,
        stt_expose_quota=32, start_size=4, recent_size=128,
        kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75,
        bool_live_gather=True, bool_hot_compaction=use_compaction)
    capture = PromptQueryCapture(model, policy)
    try:
        with torch.inference_mode():
            output = generate_with_policy(model, tokenizer, prompt, 12,
                policy, 'cuda', max_ctx=ctx)
    finally:
        capture.close()
    return output, policy


def check(a, b):
    assert a.history == b.history
    fields = ('sram_k', 'sram_v', 'sram_pos', 'sram_cum', 'sram_age',
              'sram_grace', 'stt_k', 'stt_v', 'stt_pos', 'stt_last_access',
              'stt_shadow', 'stt_was_vram_resident', 'stt_peek_streak')
    for i in a.layer_caches:
        x, y = a.layer_caches[i], b.layer_caches[i]
        for field in fields:
            left, right = getattr(x, field), getattr(y, field)
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            else:
                assert left == right, (i, field)
        assert x.metrics.summary() == y.metrics.summary(), i


def main():
    model_path = '../models/mistral-7b-instruct'
    prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    records = []
    for ctx in (4096, 8192, 16384):
        output_a, policy_a = run(model, tokenizer, prompt, ctx, False)
        output_b, policy_b = run(model, tokenizer, prompt, ctx, True)
        assert output_a == output_b, ctx
        check(policy_a, policy_b)
        records.append({'context': ctx, 'prediction_exact': True,
                        'selection_steps_exact': len(policy_a.history),
                        'cache_exact': True})
        print(ctx, 'exact', flush=True)
    atomic_json('results/baseline_fidelity/bool_hot_compaction_parity.json',
        {'config': {'runtime': runtime_identity(), 'max_gen': 12,
                    'prompt_file': 'results/remediation_v3/latency_prompt.txt'},
         'results': records})


if __name__ == '__main__':
    main()
