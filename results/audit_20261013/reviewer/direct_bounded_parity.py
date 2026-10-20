"""Reproduce the independently observed tiny-model GPU Direct parity check.

Run later from the repository root:
CUDA_VISIBLE_DEVICES=1 ../venv_kvcache/bin/python results/audit_20261013/reviewer/direct_bounded_parity.py

Saving this file did not rerun the GPU check. See independent_review.json for
the original agent-observed output and its exact scope.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / name) for name in ('experiments', 'src', 'tests')]

import torch
from transformers import MistralConfig, MistralForCausalLM
from test_october_audit import FixedTokenizer
from longbench_eval import TieredKVPolicy, generate_with_policy
from direct_shared_pool import DirectScorePolicy


def main():
    assert torch.cuda.device_count() == 1, 'Select only the authorized GPU with CUDA_VISIBLE_DEVICES'
    torch.cuda.set_device(0)
    torch.manual_seed(127)
    config = MistralConfig(num_hidden_layers=2, hidden_size=512,
        intermediate_size=768, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=128, max_position_embeddings=4096, sliding_window=None,
        attn_implementation='eager')
    model = MistralForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
    checks = []
    for hot, cold in [(8, 16), (32, 32)]:
        policies = [kind(sram_budget=hot, stt_budget=cold, start_size=1,
            recent_size=2, prefill_chunk_size=4, kv_dtype=torch.bfloat16)
            for kind in (TieredKVPolicy, DirectScorePolicy)]
        predictions, states = [], []
        try:
            for policy in policies:
                prediction = generate_with_policy(model, FixedTokenizer(),
                    'prompt', 3, policy, 'cuda')
                torch.cuda.synchronize()
                predictions.append(prediction)
                states.append({layer: (list(cache.sram_pos), list(cache.stt_pos),
                    cache.sram_k.detach().cpu().clone(),
                    cache.stt_k.detach().cpu().clone())
                    for layer, cache in policy.layer_caches.items()})
                assert all(cache.metrics.prompt_len == 20
                    for cache in policy.layer_caches.values())
            assert predictions[0] == predictions[1], predictions
            for layer in states[0]:
                assert states[0][layer][:2] == states[1][layer][:2]
                for left, right in zip(states[0][layer][2:], states[1][layer][2:]):
                    assert torch.equal(left, right)
            checks.append(dict(hot=hot, cold=cold, prediction=predictions[0], parity=True))
        finally:
            for policy in policies:
                policy.reset()
                policy.reset()
            assert policies[1].manager is None
    print(json.dumps(dict(device='CUDA_VISIBLE_DEVICES=1', passed=True,
        checks=checks, peak_allocated_bytes=torch.cuda.max_memory_allocated()), indent=2))


if __name__ == '__main__':
    main()
