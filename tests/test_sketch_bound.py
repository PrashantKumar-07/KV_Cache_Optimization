"""
test_sketch_bound.py
--------------------
Property test for the Quest page sketch.

The sketch exists to cheaply decide which STT-RAM pages are worth promoting.
That only works if it is a genuine UPPER BOUND on the largest query-key dot
product in the page: an underestimate silently discards the most relevant
page, and no downstream logic can recover from it.

The original implementation reduced over the feature dimension BEFORE taking
the maximum, computing max(sum_d q*min, sum_d q*max) instead of
sum_d max(q*min, q*max). For a query with mixed signs the two inner sums
cancel, so the result can fall far below the true page maximum -- it failed
on 60.9% of random trials. These tests pin the correct property so that
reduction order cannot regress.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tiered_kv_cache import TieredKVCache, TieredConfig
from baselines import Quest

torch.manual_seed(0)


def _quest_bound(q_hd, min_k, max_k):
    """The bound as Quest defines it: per-dimension max, then reduce over D."""
    return torch.maximum(q_hd * min_k, q_hd * max_k).sum(-1)


def test_reference_formula_is_an_upper_bound():
    """Sanity-check the property itself on raw tensors before testing callers."""
    worst = 0.0
    for _ in range(2000):
        D, P = 8, 16
        q = torch.randn(D)
        page = torch.randn(P, D)
        true_max = (page @ q).max().item()
        bound = _quest_bound(q, page.min(0).values, page.max(0).values).item()
        assert bound >= true_max - 1e-4, f"bound {bound} < true max {true_max}"
        worst = max(worst, true_max - bound)
    print(f"  [ok] reference bound holds over 2000 random pages "
          f"(worst shortfall {worst:.2e})")


def test_sum_before_max_is_not_a_bound():
    """Guard against reintroducing the old reduction order.

    If this test ever starts failing, someone has made the broken form
    correct by accident -- far more likely, the test itself was changed.
    """
    violations = 0
    for _ in range(2000):
        D, P = 8, 16
        q = torch.randn(D)
        page = torch.randn(P, D)
        true_max = (page @ q).max().item()
        mn, mx = page.min(0).values, page.max(0).values
        broken = torch.maximum((q * mn).sum(), (q * mx).sum()).item()
        if broken < true_max - 1e-5:
            violations += 1
    assert violations > 100, (
        "sum-then-max should violate the bound on a large fraction of random "
        f"mixed-sign queries; saw only {violations}/2000"
    )
    print(f"  [ok] the old sum-then-max form is confirmed broken "
          f"({violations}/2000 violations) -- regression guard is live")


def test_tiered_cache_page_bounds_are_upper_bounds():
    """TieredKVCache._page_bounds + sketch_check's scoring, on real state."""
    cfg = TieredConfig(num_heads=4, head_dim=8, sink_size=2, window_size=4,
                       sram_capacity=16, sttram_capacity=64, page_size=8)
    cache = TieredKVCache(cfg)
    N = 120
    cache.initial_bifurcation(torch.randn(4, N, 8), torch.randn(4, N, 8),
                              torch.randn(4, N, 8))

    checked = 0
    for _ in range(200):
        q = torch.randn(4, 1, 8) * 2.0        # scaled up to widen the spread
        pages = cache._page_bounds()
        if not pages:
            break
        for (start, end, min_k, max_k) in pages:
            live = [i for i in range(start, end) if not cache.stt_shadow[i]]
            if not live:
                continue
            # Call the PRODUCTION formula, not a local reimplementation:
            # a test that recomputes the bound itself passes even when
            # sketch_check regresses (verified by mutation).
            bound = TieredKVCache.page_upper_bounds(
                q, [(start, end, min_k, max_k)]
            )[0].item()
            true_max = torch.matmul(
                q, cache.stt_k[:, live, :].transpose(-2, -1)
            ).squeeze(1).sum(0).max().item()
            assert bound >= true_max - 1e-3, (
                f"page [{start},{end}) bound {bound:.4f} < true max {true_max:.4f}"
            )
            checked += 1
    assert checked > 0, "no pages were scored -- test exercised nothing"
    print(f"  [ok] TieredKVCache page bounds hold over {checked} page/query pairs")


def test_quest_baseline_scores_are_upper_bounds():
    """The same property for the standalone Quest baseline's _sketch_score."""
    q_cache = Quest(page_size=8, attend_top_pages=2, sink_size=2, window_size=4)
    N = 64
    q_cache.reset_prompt(torch.randn(4, N, 8), torch.randn(4, N, 8))

    for _ in range(100):
        q = torch.randn(4, 1, 8) * 2.0
        scores, num_pages = q_cache._sketch_score(q)
        for pi in range(num_pages):
            s, e = pi * q_cache.page_size, min((pi + 1) * q_cache.page_size, N)
            true_max = torch.matmul(
                q, q_cache.k[:, s:e, :].transpose(-2, -1)
            ).squeeze(1).sum(0).max().item()
            assert scores[pi].item() >= true_max - 1e-3, (
                f"Quest page {pi} score {scores[pi].item():.4f} "
                f"< true max {true_max:.4f}"
            )
    print("  [ok] Quest baseline page scores are upper bounds")


if __name__ == "__main__":
    test_reference_formula_is_an_upper_bound()
    test_sum_before_max_is_not_a_bound()
    test_tiered_cache_page_bounds_are_upper_bounds()
    test_quest_baseline_scores_are_upper_bounds()
    print("\nAll sketch-bound tests passed.")
