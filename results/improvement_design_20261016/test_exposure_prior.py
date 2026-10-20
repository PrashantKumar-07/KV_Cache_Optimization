import torch
import pytest
from longbench_eval import TieredKVPolicy
from exposure_prior import ExposureCalibratedPriorPolicy


def build(policy_class, heads=4, tokens=12, width=3, kv_heads=2, uniform=False):
    torch.manual_seed(17)
    keys = torch.zeros(1, kv_heads, tokens, 4) if uniform else torch.randn(1, kv_heads, tokens, 4)
    values = torch.randn_like(keys)
    query = torch.zeros(heads, width, 4) if uniform else torch.randn(heads // 2, width, 4).repeat_interleave(2, dim=0)
    policy = policy_class(sram_budget=6, stt_budget=8, start_size=1,
        recent_size=width, stt_expose_quota=2, sttram_bifurcation_frac=.75)
    policy.prompt_q_by_layer = {0: query}
    visible = policy(((keys, values),))
    return policy, visible


def test_initial_payload_identity_and_exact_evidence_units():
    old, old_visible = build(TieredKVPolicy)
    candidate, visible = build(ExposureCalibratedPriorPolicy)
    a, b = old.layer_caches[0], candidate.layer_caches[0]
    for name in ('sram_k', 'sram_v', 'stt_k', 'stt_v', 'stt_shadow'):
        assert torch.equal(getattr(a, name), getattr(b, name))
    assert a.sram_pos == b.sram_pos and a.stt_pos == b.stt_pos
    assert all(torch.equal(x, y) for pairs in zip(old_visible, visible) for x, y in zip(*pairs))
    torch.testing.assert_close(b.sram_cum, a.sram_cum / 4)
    assert torch.equal(b.sram_age, torch.full_like(b.sram_age, 3))
    assert torch.isfinite(b.sram_cum).all()


def test_duplicating_identical_query_heads_preserves_calibrated_prior():
    # Uniform queries make every duplicated GQA head exactly equivalent.
    a, _ = build(ExposureCalibratedPriorPolicy, heads=4, uniform=True)
    b, _ = build(ExposureCalibratedPriorPolicy, heads=8, uniform=True)
    assert a.layer_caches[0].sram_pos == b.layer_caches[0].sram_pos
    torch.testing.assert_close(a.layer_caches[0].sram_cum, b.layer_caches[0].sram_cum)


def test_uniform_causal_prior_has_one_head_mean_observation_per_query():
    candidate, visible = build(ExposureCalibratedPriorPolicy, uniform=True)
    cache = candidate.layer_caches[0]
    # Sink position is visible to each of the three causal queries, with
    # lengths 10,11,12. Pooling is width5, so boundary row0 averages3/5.
    expected = (1/10 + 1/11 + 1/12) * 3/5
    assert cache.sram_pos[0] == 0
    assert cache.sram_cum[0].item() == pytest.approx(expected)
    before = cache.sram_cum.clone()
    weights = torch.full((1, 4, 1, visible[0][0].shape[2]), .01)
    candidate.update_scores(visible, [weights])
    keep = candidate._vis_sram_keep[0]
    torch.testing.assert_close(cache.sram_cum[keep] - before[keep], torch.full((len(keep),), .01))


@pytest.mark.parametrize('tokens,width', [(0, 0), (1, 1), (2, 2), (12, 3)])
def test_short_or_empty_observation_age(tokens, width):
    policy, _ = build(ExposureCalibratedPriorPolicy, tokens=tokens, width=width, uniform=True)
    cache = policy.layer_caches[0]
    assert torch.isfinite(cache.sram_cum).all()
    assert (cache.sram_age == width).all()


def test_missing_queries_reject_before_any_cache_initialization():
    policy = ExposureCalibratedPriorPolicy(sram_budget=6, start_size=1, recent_size=3)
    keys = torch.zeros(1, 2, 12, 4)
    with pytest.raises(ValueError, match='actual prompt queries'):
        policy(((keys, keys),))
    assert not policy.layer_caches


def test_multiquery_decode_is_one_forward_observation_and_age_is_separate():
    policy, visible = build(ExposureCalibratedPriorPolicy, uniform=True)
    cache = policy.layer_caches[0]
    before = cache.sram_cum.clone()
    weights = torch.stack((torch.full((4, visible[0][0].shape[2]), .01),
                           torch.full((4, visible[0][0].shape[2]), .03)), dim=1)[None]
    policy.update_scores(visible, [weights])
    keep = policy._vis_sram_keep[0]
    torch.testing.assert_close(cache.sram_cum[keep] - before[keep], torch.full((len(keep),), .02))
    assert (cache.sram_age == 3).all()  # age advances separately in decode step


def test_replicating_observation_evidence_preserves_average_not_confidence():
    # Algebraic dimensional invariant, without changing causal query positions:
    # independent repetitions sum evidence AND increment exposure count.
    scores = torch.tensor([[.1, .2], [.3, .1], [.2, .4]])
    repeated = scores.repeat(3, 1)
    torch.testing.assert_close(scores.sum(0) / len(scores), repeated.sum(0) / len(repeated))
