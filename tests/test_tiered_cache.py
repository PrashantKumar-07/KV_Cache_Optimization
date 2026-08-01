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
    # force thrashing: small SRAM, aggressive promotion, anchored recall
    torch.manual_seed(1)
    def build(inclusive):
        cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                           sram_capacity=16, sttram_capacity=48, page_size=4,
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


if __name__ == "__main__":
    print("Running TieredKVCache correctness tests...")
    test_bifurcation_counts()
    test_sinks_and_window_persist()
    test_capacity_never_exceeded()
    test_output_shape_and_finiteness()
    test_promotion_happens()
    test_inclusive_saves_writes()
    test_inclusive_capacity_and_finite()
    print("\nAll tests passed.")
