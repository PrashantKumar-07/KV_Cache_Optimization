"""
test_baselines.py
-----------------
Invariants for the standalone baseline cache policies in src/baselines.py.

These policies are what TieredKV's accuracy is argued against, so a silently
crippled baseline is as damaging to the claim as a broken TieredKV. Two real
defects motivated this file, both of which would have been caught here:

  1. H2O.reset_prompt kept only sinks + window, discarding the rest of its
     budget: at budget=1024 on a 4000-token prompt it held 20 tokens.
  2. H2O.step / SnapKV.step called argmin on an all-inf candidate vector
     whenever budget < sink_size + window_size. argmin returns index 0 there,
     which is the first attention sink, so the sinks were evicted first.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from baselines import FullAttention, StreamingLLM, H2O, SnapKV, Quest

torch.manual_seed(0)

H, D = 2, 8


def _prompt(N):
    return torch.randn(H, N, D), torch.randn(H, N, D), torch.randn(H, N, D)


def _decode(policy, steps, start_pos):
    for s in range(steps):
        out = policy.step(torch.randn(H, 1, D), torch.randn(H, 1, D),
                          torch.randn(H, 1, D), start_pos + s)
        assert out.shape == (H, 1, D), f"bad output shape {out.shape}"
        assert torch.isfinite(out).all(), "non-finite attention output"
    return policy


def test_h2o_reset_prompt_uses_its_whole_budget():
    """H2O must seed with `budget` tokens, not sink+window."""
    N, budget = 400, 128
    k, v, q = _prompt(N)
    h = H2O(budget=budget, sink_size=4, window_size=16)
    h.reset_prompt(k, v, q)
    assert len(h.pos) == budget, (
        f"H2O kept {len(h.pos)} of a {budget}-token budget "
        f"(sink+window would be {4 + 16})"
    )
    # and it must still pin the sinks and the recent window
    for s in range(4):
        assert s in h.pos, f"sink {s} missing after reset_prompt"
    assert N - 1 in h.pos, "most recent prompt token missing after reset_prompt"
    print(f"  [ok] H2O.reset_prompt fills its budget ({len(h.pos)}/{budget}) "
          f"and pins sinks + window")


def test_snapkv_reset_prompt_uses_its_whole_budget():
    N, budget = 400, 128
    k, v, q = _prompt(N)
    s = SnapKV(budget=budget, sink_size=4, window_size=16)
    s.reset_prompt(k, v, q)
    assert len(s.pos) == budget, f"SnapKV kept {len(s.pos)} of {budget}"
    print(f"  [ok] SnapKV.reset_prompt fills its budget ({len(s.pos)}/{budget})")


def test_sinks_survive_when_budget_is_smaller_than_sink_plus_window():
    """The degenerate configuration that used to wipe positions 0-3."""
    N = 50
    k, v, q = _prompt(N)
    for cls in (H2O, SnapKV):
        p = cls(budget=16, sink_size=4, window_size=16)   # 16 < 4 + 16
        p.reset_prompt(k, v, q)
        _decode(p, 10, N)
        missing = [s for s in range(4) if s not in p.pos]
        assert not missing, f"{cls.__name__} evicted attention sinks {missing}"
    print("  [ok] H2O and SnapKV keep their sinks even when "
          "budget < sink_size + window_size")


def test_budget_is_respected_when_it_is_satisfiable():
    """With budget >= sink+window the policies must hold the budget exactly."""
    N = 200
    k, v, q = _prompt(N)
    for cls in (H2O, SnapKV):
        p = cls(budget=64, sink_size=4, window_size=16)
        p.reset_prompt(k, v, q)
        _decode(p, 60, N)
        assert p.num_tokens <= 64, f"{cls.__name__} overshot budget: {p.num_tokens}"
        for s in range(4):
            assert s in p.pos, f"{cls.__name__} evicted sink {s}"
    print("  [ok] H2O and SnapKV hold their budget across 60 decode steps")


def test_streamingllm_holds_sink_plus_window():
    N = 200
    k, v, _ = _prompt(N)
    p = StreamingLLM(sink_size=4, window_size=32)
    p.reset_prompt(k, v)
    _decode(p, 60, N)
    assert p.num_tokens <= 4 + 32, f"StreamingLLM overshot: {p.num_tokens}"
    for s in range(4):
        assert s in p.pos, f"StreamingLLM evicted sink {s}"
    print(f"  [ok] StreamingLLM holds sink+window ({p.num_tokens} tokens)")


def test_full_attention_keeps_everything():
    N = 64
    k, v, _ = _prompt(N)
    p = FullAttention()
    p.reset_prompt(k, v)
    _decode(p, 20, N)
    assert p.num_tokens == N + 20, f"FullAttention dropped tokens: {p.num_tokens}"
    print(f"  [ok] FullAttention retains all {p.num_tokens} tokens")


def test_quest_keeps_full_cache_and_bounds_attended_set():
    """Quest evicts nothing; it only sparsifies which pages are attended."""
    N = 64
    k, v, _ = _prompt(N)
    p = Quest(page_size=8, attend_top_pages=2, sink_size=4, window_size=8)
    p.reset_prompt(k, v)
    _decode(p, 20, N)
    assert p.num_tokens == N + 20, f"Quest evicted tokens: {p.num_tokens}"
    assert p.total_sketch_macs > 0, "Quest never scored a page"
    print(f"  [ok] Quest keeps the full cache ({p.num_tokens}) and sketches pages")


if __name__ == "__main__":
    test_h2o_reset_prompt_uses_its_whole_budget()
    test_snapkv_reset_prompt_uses_its_whole_budget()
    test_sinks_survive_when_budget_is_smaller_than_sink_plus_window()
    test_budget_is_respected_when_it_is_satisfiable()
    test_streamingllm_holds_sink_plus_window()
    test_full_attention_keeps_everything()
    test_quest_keeps_full_cache_and_bounds_attended_set()
    print("\nAll baseline tests passed.")
