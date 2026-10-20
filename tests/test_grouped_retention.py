"""Actual model parity and head/payload identity for the opt-in ablation."""
from pathlib import Path
import sys
import pytest
import torch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/name) for name in ('experiments','src')]
from grouped_retention import GroupedRetentionPolicy
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy, generate_with_policy
from test_longbench_policies import _tiny_model
from test_october_audit import FixedTokenizer


def test_no_drop_grouped_head_model_logits_and_kv_match_global():
    model = _tiny_model()
    records = []
    for kind in (TieredKVPolicy, GroupedRetentionPolicy):
        policy = kind(sram_budget=32, stt_budget=32, start_size=1, recent_size=2)
        capture = PromptQueryCapture(model, policy)
        logits = []
        handle = model.register_forward_hook(lambda m,args,out: logits.append(out.logits.clone()))
        try:
            generated = generate_with_policy(model, FixedTokenizer(), 'prompt', 3, policy, 'cpu')
        finally:
            handle.remove(); capture.close()
        records.append((generated, logits, policy))
    left, right = records
    assert left[0] == right[0]
    for a,b in zip(left[1], right[1]):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    for layer, cache in left[2].layer_caches.items():
        for head in range(cache.cfg.num_heads):
            grouped = right[2].layer_caches[layer*cache.cfg.num_heads+head]
            assert grouped.sram_pos == cache.sram_pos
            torch.testing.assert_close(grouped.sram_k, cache.sram_k[head:head+1])
            torch.testing.assert_close(grouped.sram_v, cache.sram_v[head:head+1])


def test_distinct_head_queries_retain_distinct_rows_with_aligned_values():
    keys = torch.zeros(1,2,20,2)
    keys[0,0,2:5,0] = 8
    keys[0,1,12:15,1] = 8
    values = torch.arange(40).reshape(1,2,20,1).expand_as(keys).float()
    queries = torch.zeros(4,2,2); queries[:2,:,0] = 1; queries[2:,:,1] = 1
    policy = GroupedRetentionPolicy(sram_budget=4, stt_budget=8, start_size=1,
                                    recent_size=2, stt_expose_quota=2)
    policy.prompt_q_by_layer = {0:queries}
    visible = policy(((keys,values),))
    assert visible[0][0].shape[1] == 2
    assert set(policy.layer_caches[0].sram_pos) & set(range(2,5))
    assert set(policy.layer_caches[1].sram_pos) & set(range(12,15))
    for head, cache in policy.layer_caches.items():
        for positions, payload in [(cache.sram_pos,cache.sram_v),(cache.stt_pos,cache.stt_v)]:
            torch.testing.assert_close(payload, values[0,head:head+1,positions])
        assert len(cache.sram_pos) <= 4 and len(cache.stt_pos) <= 8
    assert not policy.prompt_q_by_layer


def test_missing_query_group_rejects_instead_of_proxy_fallback():
    policy = GroupedRetentionPolicy(sram_budget=4, start_size=1, recent_size=1)
    keys = torch.zeros(1,2,8,2)
    with pytest.raises(ValueError, match='every model layer'):
        policy(((keys,keys),))


def test_bounded_prefill_combination_rejects():
    with pytest.raises(ValueError, match='full prefill'):
        GroupedRetentionPolicy(prefill_chunk_size=4)


def test_zero_query_heads_reject_before_cache_initialization():
    policy = GroupedRetentionPolicy(sram_budget=4, start_size=1, recent_size=2)
    keys = torch.zeros(1, 2, 8, 2)
    policy.prompt_q_by_layer = {0: torch.zeros(0, 2, 2)}
    with pytest.raises(ValueError, match='query geometry'):
        policy(((keys, keys),))
    assert not policy.layer_caches


def test_zero_feedback_heads_reject_without_corrupting_scores():
    policy = GroupedRetentionPolicy(sram_budget=4, start_size=1, recent_size=2)
    keys = torch.zeros(1, 2, 8, 2)
    policy.prompt_q_by_layer = {0: torch.zeros(4, 2, 2)}
    visible = policy(((keys, keys),))
    before = [cache.sram_cum.clone() for cache in policy.layer_caches.values()]
    with pytest.raises(ValueError, match='GQA feedback'):
        policy.update_scores(visible, [torch.zeros(1, 0, 1, visible[0][0].shape[2])])
    for cache, scores in zip(policy.layer_caches.values(), before):
        torch.testing.assert_close(cache.sram_cum, scores)
        assert torch.isfinite(cache.sram_cum).all()
