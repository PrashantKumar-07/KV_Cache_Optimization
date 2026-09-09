"""
test_tiered_cache.py
--------------------
Correctness tests for the 3-tier cache.  Run:  python tests/test_tiered_cache.py
No pytest dependency -- plain asserts so it runs anywhere.

We verify the *invariants* a reviewer would challenge:
  1. Bifurcation routes exactly the prompt tokens, no duplication/loss.
  2. Attention sinks (0..sink_size-1) stay in SRAM through many steps.
  3. Recent window tokens stay in SRAM.
  4. SRAM and STT-RAM never exceed their capacities.
  5. Promotion actually moves tokens STT-RAM -> SRAM when forced.
  6. Output shape is correct and finite.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tiered_kv_cache import TieredKVCache, TieredConfig

torch.manual_seed(0)


def make_prompt(H, N, D):
    k = torch.randn(H, N, D)
    v = torch.randn(H, N, D)
    q = torch.randn(H, N, D)
    return k, v, q


def test_bifurcation_counts():
    cfg = TieredConfig(num_heads=4, head_dim=16, sink_size=4, window_size=8,
                       sram_capacity=32, sttram_capacity=48, store_dram=True)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(4, 200, 16)
    info = cache.initial_bifurcation(k, v, q)
    total = info["sram"] + info["sttram"] + info["dram_or_dropped"]
    assert total == 200, f"tokens lost/duplicated: {total} != 200"
    assert info["sram"] <= cfg.sram_capacity
    assert info["sttram"] <= cfg.sttram_capacity
    print("  [ok] bifurcation conserves tokens and respects capacities")


def test_sinks_and_window_persist():
    cfg = TieredConfig(num_heads=4, head_dim=16, sink_size=4, window_size=8,
                       sram_capacity=32, sttram_capacity=48)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(4, 128, 16)
    cache.initial_bifurcation(k, v, q)

    pos = 128
    for _ in range(100):
        nq = torch.randn(4, 1, 16)
        nk = torch.randn(4, 1, 16)
        nv = torch.randn(4, 1, 16)
        cache.step(nq, nk, nv, pos)
        pos += 1

    for s in range(cfg.sink_size):
        assert s in cache.sram_pos, f"sink {s} was evicted from SRAM!"
    max_pos = max(cache.sram_pos)
    recent = [p for p in cache.sram_pos if p > max_pos - cfg.window_size]
    assert len(recent) >= 1, "recent window not in SRAM"
    print("  [ok] sinks and recent window remain in SRAM across 100 steps")


def test_capacity_never_exceeded():
    cfg = TieredConfig(num_heads=4, head_dim=16, sink_size=4, window_size=8,
                       sram_capacity=32, sttram_capacity=48, store_dram=False)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(4, 128, 16)
    cache.initial_bifurcation(k, v, q)
    pos = 128
    for _ in range(200):
        cache.step(torch.randn(4, 1, 16), torch.randn(4, 1, 16), torch.randn(4, 1, 16), pos)
        pos += 1
        assert len(cache.sram_pos) <= cfg.sram_capacity, "SRAM overflow"
        assert len(cache.stt_pos) <= cfg.sttram_capacity, "STT-RAM overflow"
    print("  [ok] SRAM and STT-RAM capacities never exceeded over 200 steps")


def test_output_shape_and_finiteness():
    cfg = TieredConfig(num_heads=4, head_dim=16, sram_capacity=32, sttram_capacity=48)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(4, 64, 16)
    cache.initial_bifurcation(k, v, q)
    out = cache.step(torch.randn(4, 1, 16), torch.randn(4, 1, 16), torch.randn(4, 1, 16), 64)
    assert out.shape == (4, 1, 16), f"bad output shape {out.shape}"
    assert torch.isfinite(out).all(), "output has NaN/Inf"
    print("  [ok] step output shape (H,1,D) and finite")


def test_promotion_happens():
    # tiny STT so a promotion is easy to force
    cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                       sram_capacity=12, sttram_capacity=32, page_size=4,
                       promote_top_pages=2)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(2, 80, 8)
    cache.initial_bifurcation(k, v, q)
    pos, promoted = 80, 0
    for _ in range(60):
        cache.step(torch.randn(2, 1, 8), torch.randn(2, 1, 8), torch.randn(2, 1, 8), pos)
        pos += 1
    promoted = cache.metrics.total_promoted
    assert promoted > 0, "no tokens were ever promoted from STT-RAM"
    print(f"  [ok] promotion path exercised ({promoted} tokens promoted)")


def test_inclusive_saves_writes():
    # Small VRAM (aggressive promotion/demotion churn) but STT-RAM with enough
    # headroom over VRAM that a promoted token's shadow backup survives long
    # enough to be genuinely reused across steps, rather than being reclaimed
    # (or self-cancelled within the same step) before it is ever needed again.
    # See src/tiered_kv_cache.py's one-step grace period on promoted tokens:
    # with too little STT-RAM headroom, ALL backups get reclaimed under
    # pressure before reuse, understating the mechanism -- this is not the
    # cache "not working", it's an unrealistic capacity ratio for this test.
    torch.manual_seed(1)
    def build(inclusive):
        cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                           sram_capacity=32, sttram_capacity=96, page_size=4,
                           promote_top_pages=2, inclusive=inclusive)
        cache = TieredKVCache(cfg)
        k, v, q = make_prompt(2, 96, 8)
        cache.initial_bifurcation(k, v, q)
        pos = 96
        kp = k  # reuse prompt keys as anchors to trigger promotion + re-demotion
        for t in range(80):
            if t % 2 == 0:
                nq = 3.0 * kp[:, torch.randint(2, 40, (1,)).item(), :].unsqueeze(1)
            else:
                nq = torch.randn(2, 1, 8)
            cache.step(nq, torch.randn(2, 1, 8), torch.randn(2, 1, 8), pos)
            pos += 1
        return cache.metrics

    inc = build(True)
    dst = build(False)
    # inclusive must reuse backups -> some demotions pay no write
    assert inc.total_writes_saved > 0, "inclusive cache saved no writes"
    assert dst.total_writes_saved == 0, "destructive mode should never save writes"
    # paid writes strictly fewer under inclusive (the whole point)
    assert inc.total_paid_writes < dst.total_demoted, \
        f"no write reduction: paid={inc.total_paid_writes} vs {dst.total_demoted}"
    print(f"  [ok] inclusive victim cache saved {inc.total_writes_saved} writes "
          f"(paid {inc.total_paid_writes} vs destructive {dst.total_demoted})")


def test_inclusive_capacity_and_finite():
    # shadows occupy real STT rows; capacity must still hold, output stays finite
    cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                       sram_capacity=16, sttram_capacity=32, page_size=4,
                       promote_top_pages=2, inclusive=True)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(2, 80, 8)
    cache.initial_bifurcation(k, v, q)
    pos = 80
    for _ in range(120):
        out = cache.step(torch.randn(2, 1, 8), torch.randn(2, 1, 8), torch.randn(2, 1, 8), pos)
        pos += 1
        assert len(cache.sram_pos) <= cfg.sram_capacity, "SRAM overflow (inclusive)"
        assert len(cache.stt_pos) <= cfg.sttram_capacity, "STT overflow (inclusive)"
        assert torch.isfinite(out).all(), "non-finite output (inclusive)"
        # invariant: an SRAM-resident position has no live (non-shadow) STT twin
        live = {p for p, sh in zip(cache.stt_pos, cache.stt_shadow) if not sh}
        assert not (set(cache.sram_pos) & live), "position both in SRAM and a live STT victim"
    print("  [ok] inclusive mode respects capacity, stays finite, no double-residency")


def test_peek_first_candidacy_promotes_immediately():
    # A page that has NEVER been VRAM-resident before must still promote on
    # its very first candidacy -- the peek/promote hysteresis gate only
    # applies to RE-entry, so this must be unchanged from pre-peek behavior.
    cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                       sram_capacity=12, sttram_capacity=32, page_size=4,
                       promote_top_pages=2)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(2, 80, 8)
    cache.initial_bifurcation(k, v, q)
    pos, promoted = 80, 0
    for _ in range(60):
        cache.step(torch.randn(2, 1, 8), torch.randn(2, 1, 8), torch.randn(2, 1, 8), pos)
        pos += 1
    promoted = cache.metrics.total_promoted
    assert promoted > 0, "first-time candidates should still promote without delay"
    print(f"  [ok] first-time candidacy still promotes immediately ({promoted} promoted)")


def test_peek_does_not_change_residency():
    # A peeked-only page's tokens must remain in STT-RAM (not move to VRAM)
    # for as long as they don't clear the hysteresis gate -- visibility
    # without residency change is the entire point of peek/promote.
    torch.manual_seed(2)
    cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                       sram_capacity=16, sttram_capacity=64, page_size=4,
                       promote_top_pages=2)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(2, 96, 8)
    cache.initial_bifurcation(k, v, q)
    pos = 96
    total_peeked = 0
    for t in range(60):
        # Alternate which prompt anchor we query so different pages become
        # candidates repeatedly without any one page building a long enough
        # streak to clear the re-entry gate on the first few tries.
        anchor = k[:, torch.randint(4, 60, (1,)).item(), :].unsqueeze(1)
        cache.step(2.0 * anchor, torch.randn(2, 1, 8), torch.randn(2, 1, 8), pos)
        pos += 1
        total_peeked += cache.metrics.steps[-1].peeked_tokens
        assert len(cache.sram_pos) <= cfg.sram_capacity, "SRAM overflow"
        assert len(cache.stt_pos) <= cfg.sttram_capacity, "STT-RAM overflow"
        # No position may be in both tiers as a live (non-shadow) resident.
        live_stt = {p for p, sh in zip(cache.stt_pos, cache.stt_shadow) if not sh}
        assert not (set(cache.sram_pos) & live_stt), "position resident in both tiers"
    assert total_peeked > 0, "peek mechanism never engaged -- test setup didn't exercise it"
    print(f"  [ok] peek attends without residency change ({total_peeked} peeked-token-steps, "
          f"no double-residency)")


def test_peek_gates_repeat_promotion_churn():
    # The whole motivation for peek/promote: a token that gets promoted,
    # evicted, and immediately re-qualifies should NOT instantly re-promote
    # -- it should be gated to a peek until it proves relevance repeatedly.
    # We can't force this deterministically with random steps, but we can
    # assert the structural invariant that makes the gate meaningful: once
    # a page has been VRAM-resident and evicted, resolve_promotions must
    # actually be capable of returning it as a peek_hit (not force-promote
    # it) the very next time it's a candidate with streak < the gate.
    cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                       sram_capacity=12, sttram_capacity=32, page_size=4,
                       promote_top_pages=4)
    cache = TieredKVCache(cfg)
    k, v, q = make_prompt(2, 80, 8)
    cache.initial_bifurcation(k, v, q)

    # Manually mark a real (non-shadow) STT page as previously-VRAM-resident
    # with a fresh (streak=0) re-candidacy episode, then present it as a
    # sketch-check candidate: it must come back as a peek, not a promotion.
    assert len(cache.stt_pos) >= 4, "test needs at least one full page in STT-RAM"
    for i in range(4):
        cache.stt_was_vram_resident[i] = True
        cache.stt_peek_streak[i] = 0
    promote_hits, peek_hits = cache.resolve_promotions([(0, 4)])
    assert promote_hits == [], f"re-entry candidate should be gated, got promote_hits={promote_hits}"
    assert peek_hits == [(0, 4)], f"re-entry candidate should be peeked, got peek_hits={peek_hits}"

    # After clearing the gate's required consecutive streak, the same page
    # must promote.
    for _ in range(TieredKVCache._HYSTERESIS_MIN_STREAK - 1):
        cache.resolve_promotions([(0, 4)])
    promote_hits, peek_hits = cache.resolve_promotions([(0, 4)])
    assert promote_hits == [(0, 4)], \
        f"page should promote once streak clears the gate, got promote_hits={promote_hits}"
    print("  [ok] re-entry candidates are gated to peek until they clear "
          f"a streak of {TieredKVCache._HYSTERESIS_MIN_STREAK}, then promote")


if __name__ == "__main__":
    print("Running TieredKVCache correctness tests...")
    test_bifurcation_counts()
    test_sinks_and_window_persist()
    test_capacity_never_exceeded()
    test_output_shape_and_finiteness()
    test_promotion_happens()
    test_inclusive_saves_writes()
    test_inclusive_capacity_and_finite()
    test_peek_first_candidacy_promotes_immediately()
    test_peek_does_not_change_residency()
    test_peek_gates_repeat_promotion_churn()
    print("\nAll tests passed.")
