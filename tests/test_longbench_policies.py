"""
test_longbench_policies.py
--------------------------
Integration tests for experiments/longbench_eval.py against a tiny randomly
initialised Mistral.

This file exists because every published number comes from longbench_eval.py
and, until now, nothing tested it. The two bugs that were found in it were
both found by inspection AFTER results had been reported:

  1. TieredKVPolicy.reset() wiped layer_caches without folding the outgoing
     sample's counters, so cache_stats reported only the LAST sample of a
     task (giveaway: decode_steps=3 on a 25-sample task).
  2. _visible_tuple exposed the entire live STT tier, so TieredKV attended
     ~3x more tokens than any equal-budget baseline.

Both are asserted here. The model is 2 layers of hidden size 64 so the whole
file runs in seconds on CPU and needs no downloaded weights.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import MistralConfig, MistralForCausalLM

import longbench_eval as LE
from longbench_eval import (
    StreamingLLMPolicy, H2OPolicy, SnapKVPolicy, FullCachePolicy,
    TieredKVPolicy, to_tuple_kv, generate_with_policy,
)

torch.manual_seed(0)

LAYERS, HEADS, KV_HEADS, DIM = 2, 4, 2, 16
HEAD_DIM = DIM // HEADS


def _tiny_model():
    cfg = MistralConfig(
        num_hidden_layers=LAYERS, hidden_size=DIM, intermediate_size=32,
        num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        vocab_size=128, max_position_embeddings=4096,
        sliding_window=None, attn_implementation="eager",
    )
    m = MistralForCausalLM(cfg)
    m.eval()
    return m


class _StubTokenizer:
    """Whitespace tokenizer -- generate_with_policy only needs ids in, text out."""
    eos_token_id = 127
    pad_token = None

    def __call__(self, text, truncation=False, return_tensors="pt"):
        ids = [(abs(hash(w)) % 100) + 3 for w in text.split()]
        class _Out: pass
        o = _Out()
        o.input_ids = torch.tensor([ids or [3]])
        return o

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _fake_past(seq_len):
    """A tuple-of-(k, v) shaped exactly like a real DynamicCache's layers."""
    return tuple(
        (torch.randn(1, KV_HEADS, seq_len, HEAD_DIM),
         torch.randn(1, KV_HEADS, seq_len, HEAD_DIM))
        for _ in range(LAYERS)
    )


# ---------------------------------------------------------------------------
# Budget invariants, per policy
# ---------------------------------------------------------------------------
def test_every_policy_respects_its_budget():
    budget, seq = 64, 300
    past = _fake_past(seq)
    for policy in (StreamingLLMPolicy(start_size=4, recent_size=budget - 4),
                   H2OPolicy(budget=budget, start_size=4, recent_size=16),
                   SnapKVPolicy(budget=budget, start_size=4, recent_size=16)):
        out = policy(past)
        for layer_idx, (k, v) in enumerate(out):
            assert k.size(2) <= budget, (
                f"{policy.name} layer {layer_idx} kept {k.size(2)} > budget {budget}"
            )
            assert k.size(2) == v.size(2), f"{policy.name} K/V length mismatch"
    print(f"  [ok] StreamingLLM, H2O and SnapKV all trim {seq} -> <= {budget}")


def test_full_policy_keeps_everything():
    past = _fake_past(300)
    out = FullCachePolicy()(past)
    assert out[0][0].size(2) == 300, "Full policy trimmed the cache"
    print("  [ok] Full policy keeps all 300 tokens")


def test_sinks_survive_repeated_eviction():
    """Positions 0..start_size-1 must stay at the front of the kept tensor."""
    budget, seq = 64, 200
    past = _fake_past(seq)
    for policy in (H2OPolicy(budget=budget, start_size=4, recent_size=16),
                   SnapKVPolicy(budget=budget, start_size=4, recent_size=16)):
        sink_ref = past[0][0][:, :, :4, :].clone()
        out = policy(past)
        kept_sinks = out[0][0][:, :, :4, :]
        assert torch.allclose(kept_sinks, sink_ref), (
            f"{policy.name} did not keep the four attention sinks at the front"
        )
    print("  [ok] H2O and SnapKV preserve the four attention sinks")


# ---------------------------------------------------------------------------
# TieredKV: uniform visible length, bounded exposure, stats accumulation
# ---------------------------------------------------------------------------
def _tiered(seq=300, vram=64, stt=128, expose=None):
    p = TieredKVPolicy(sram_budget=vram, stt_budget=stt, start_size=4,
                       recent_size=16, page_size=8, promote_top_pages=2,
                       stt_expose_quota=expose)
    return p, _fake_past(seq)


def test_visible_tuple_length_is_uniform_across_layers():
    """HF builds ONE causal mask per forward call, shared by every layer, so
    a per-layer length disagreement is a silent correctness bug."""
    p, past = _tiered()
    out = p(past)
    lengths = {k.size(2) for k, _ in out}
    assert len(lengths) == 1, f"layers disagree on visible length: {lengths}"
    for k, v in out:
        assert k.size(2) == v.size(2), "K/V length mismatch in visible tuple"
    print(f"  [ok] visible tuple is uniform across {LAYERS} layers "
          f"({lengths.pop()} tokens)")


def test_attended_pool_is_bounded_by_vram_plus_expose_quota():
    """The regression that made the headline comparison unfair.

    Without a quota the visible set grew to vram + the whole live STT tier,
    letting TieredKV attend ~3x more tokens than an equal-budget baseline.
    """
    vram, stt, expose = 64, 256, 16
    p, past = _tiered(seq=600, vram=vram, stt=stt, expose=expose)
    out = p(past)
    visible = out[0][0].size(2)
    assert visible <= vram + expose, (
        f"attended {visible} tokens, quota allows {vram} + {expose} = {vram + expose}"
    )
    assert visible > vram, "slow tier contributed nothing -- exposure is broken"
    print(f"  [ok] attended pool is {visible} tokens, bounded by "
          f"vram({vram}) + quota({expose})")


def test_default_expose_quota_is_the_sketch_quantum():
    p = TieredKVPolicy(sram_budget=64, stt_budget=128, page_size=8,
                       promote_top_pages=2)
    assert p.stt_expose_quota == 16, p.stt_expose_quota
    print(f"  [ok] default expose quota = promote_top_pages*page_size = "
          f"{p.stt_expose_quota}")


def test_stats_accumulate_across_samples_not_just_the_last():
    """reset() must fold the outgoing sample's counters into _accum.

    The pre-fix version reported only whatever survived in layer_caches at
    stats() time, i.e. the final sample of a task.
    """
    p, past = _tiered()
    p(past)                                    # sample 1: bifurcate
    for _ in range(3):
        p(_fake_past(p.sram_budget + 1))       # sample 1: three decode steps
    after_one = p.stats()["decode_steps"]
    assert after_one == 3, after_one

    p.reset()                                  # sample boundary
    p(past)                                    # sample 2
    for _ in range(2):
        p(_fake_past(p.sram_budget + 1))
    after_two = p.stats()["decode_steps"]
    assert after_two == 5, (
        f"decode_steps={after_two} after 3+2 steps across two samples; "
        f"reset() is discarding the first sample"
    )
    print(f"  [ok] stats accumulate across samples (3 + 2 = {after_two} steps)")


def test_stats_reports_deferred_promotions_not_phantom_peeks():
    """`peeked` used to mean 'attended in place', which this harness never does."""
    p, past = _tiered()
    p(past)
    p(_fake_past(p.sram_budget + 1))
    s = p.stats()
    assert "promotions_deferred" in s, "deferred-promotion counter missing"
    assert "peeked" not in s, "stale 'peeked' key still reported"
    assert "stt_starved_steps" in s, "slow-tier starvation counter missing"
    print("  [ok] stats report promotions_deferred and stt_starved_steps")


def test_slow_tier_starvation_is_detected_and_reported():
    """One fully-shadowed layer zeroes the exposed slow tier for ALL layers.

    _visible_tuple sizes the exposed set to the minimum live count across
    layers, so this is a single point of failure: TieredKV silently degrades
    to a VRAM-only policy, which reads as an accuracy result rather than the
    degenerate state it is. Asserted directly on _visible_tuple, because a
    full decode step repopulates the live set (evict_and_demote appends the
    demoted token) before the check is reached.
    """
    import io, contextlib
    p, past = _tiered()
    p(past)
    assert p.stats()["stt_starved_steps"] == 0, "starved before anything went wrong"

    p.layer_caches[0].stt_shadow[:] = True          # one layer, fully shadowed
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = p._visible_tuple(past)

    assert "WARNING" in buf.getvalue(), "starvation was not surfaced to the operator"
    assert p.stats()["stt_starved_steps"] == 1, p.stats()["stt_starved_steps"]
    assert len({k.size(2) for k, _ in out}) == 1, "layers disagree while starved"
    print(f"  [ok] slow-tier starvation warns and counts "
          f"(visible collapsed to {out[0][0].size(2)} VRAM-only tokens)")


def test_mean_slow_tier_exposure_is_reported():
    """The common degradation is throttling to the cross-layer minimum, not
    a hard zero -- a nominally full 2048-token tier can contribute 1 token."""
    p, past = _tiered(seq=600, vram=64, stt=256, expose=16)
    p(past)
    healthy = p.stats()["mean_stt_exposed"]
    assert healthy > 0, "no slow-tier exposure recorded at all"
    assert p.stats()["stt_expose_quota"] == 16

    # Throttle one layer to a single live token; every layer must follow it.
    p.layer_caches[0].stt_shadow[:] = True
    p.layer_caches[0].stt_shadow[0] = False
    p._visible_tuple(past)
    throttled = p.stats()["mean_stt_exposed"]
    assert throttled < healthy, (
        f"mean exposure {throttled} did not drop below {healthy} when one "
        f"layer was throttled to a single live token"
    )
    print(f"  [ok] mean slow-tier exposure tracks throttling "
          f"({healthy} -> {throttled} against a quota of 16)")


# ---------------------------------------------------------------------------
# End to end through the real generation loop
# ---------------------------------------------------------------------------
def test_generate_with_policy_runs_for_every_method():
    model, tok = _tiny_model(), _StubTokenizer()
    text = " ".join(f"w{i}" for i in range(200))
    budget = 64
    for policy in (FullCachePolicy(),
                   StreamingLLMPolicy(start_size=4, recent_size=budget - 4),
                   H2OPolicy(budget=budget, start_size=4, recent_size=16),
                   SnapKVPolicy(budget=budget, start_size=4, recent_size=16),
                   TieredKVPolicy(sram_budget=budget, stt_budget=128,
                                  start_size=4, recent_size=16, page_size=8,
                                  promote_top_pages=2)):
        policy.reset()
        out = generate_with_policy(model, tok, text, max_gen=6, policy=policy,
                                   device="cpu", max_ctx=256)
        assert isinstance(out, str), f"{policy.name} returned {type(out)}"
    print("  [ok] all five policies complete a real decode loop on a tiny Mistral")


def test_true_positions_are_passed_during_decode():
    """Every sparse policy trims the cache, so HF's default
    arange + get_seq_length() would assign false RoPE distances."""
    model, tok = _tiny_model(), _StubTokenizer()
    seen = []
    original = model.forward

    def spy(*args, **kwargs):
        if kwargs.get("position_ids") is not None:
            seen.append(int(kwargs["position_ids"].flatten()[-1]))
        return original(*args, **kwargs)

    model.forward = spy
    try:
        policy = StreamingLLMPolicy(start_size=4, recent_size=28)
        text = " ".join(f"w{i}" for i in range(120))
        generate_with_policy(model, tok, text, max_gen=5, policy=policy,
                             device="cpu", max_ctx=256)
    finally:
        model.forward = original

    decode_positions = seen[1:]          # drop the prefill call
    assert decode_positions == sorted(decode_positions), decode_positions
    assert decode_positions[0] > 32, (
        f"first decode position {decode_positions[0]} looks like a trimmed "
        f"cache length, not a true document position"
    )
    assert all(b - a == 1 for a, b in zip(decode_positions, decode_positions[1:])), \
        decode_positions
    print(f"  [ok] decode positions are true and contiguous: {decode_positions}")


def test_quest_policy_refuses_to_run_rather_than_faking_a_score():
    try:
        LE.QuestPolicy()
    except NotImplementedError:
        print("  [ok] QuestPolicy raises instead of silently reporting Full-KV")
        return
    raise AssertionError("QuestPolicy should raise NotImplementedError")


def test_reset_does_not_double_count_starvation_counters():
    """Regression: reset() folded _live_totals() (which already includes the
    stt_* counters) and then added the same three counters AGAIN — ~2x
    inflated starvation/exposure stats in every result file."""
    p, past = _tiered()
    p(past)                                    # bifurcate so layer_caches exist
    p._stt_starved_steps = 3
    p._stt_exposed_total = 30
    p._stt_exposure_samples = 5
    p.reset()
    assert p._accum["stt_starved_steps"] == 3, p._accum["stt_starved_steps"]
    assert p._accum["stt_exposed_total"] == 30, p._accum["stt_exposed_total"]
    assert p._accum["stt_exposure_samples"] == 5, p._accum["stt_exposure_samples"]
    # Second sample boundary must add exactly once more, not compound.
    p(past)
    p._stt_starved_steps = 3
    p._stt_exposed_total = 30
    p._stt_exposure_samples = 5
    p.reset()
    assert p._accum["stt_starved_steps"] == 6, p._accum["stt_starved_steps"]
    assert p._accum["stt_exposed_total"] == 60, p._accum["stt_exposed_total"]
    assert p._accum["stt_exposure_samples"] == 10, p._accum["stt_exposure_samples"]
    print("  [ok] reset() counts starvation/exposure exactly once per sample")


def test_dropped_tokens_bill_drop_cost():
    """Dropped tokens are freed, not migrated: the harness must bill
    drop_cost() (one retiring STT read) unconditionally."""
    import inspect
    from cost_model import CostModel
    cm = CostModel()
    n = 1000
    lat_drop, eng_drop = cm.drop_cost(n)
    assert abs(lat_drop - cm.read_latency_us("STT-RAM", n)) < 1e-9
    assert abs(eng_drop - cm.read_energy_nj("STT-RAM", n)) < 1e-9
    src = inspect.getsource(LE.TieredKVPolicy._decode_step)
    assert "drop_cost" in src, "harness _decode_step stopped billing drop_cost"
    assert "store_dram" not in src and "deep_demote" not in src, (
        "dead DRAM-tier references are back in _decode_step"
    )
    print(f"  [ok] drop_cost ({lat_drop:.1f}us) billed unconditionally, "
          f"no DRAM-tier references in _decode_step")


def test_live_floor_reaches_layer_caches():
    """The starvation guard must actually arrive in TieredConfig, or the
    default CLI value silently runs unfixed code."""
    p, past = _tiered()
    assert p.stt_live_floor == 1, p.stt_live_floor
    p(past)                                    # bifurcate
    for c in p.layer_caches.values():
        assert c.cfg.stt_live_floor == 1, c.cfg.stt_live_floor
    p2 = TieredKVPolicy(sram_budget=64, stt_budget=128, start_size=4,
                        recent_size=16, page_size=8, promote_top_pages=2,
                        stt_live_floor=5)
    p2(past)
    for c in p2.layer_caches.values():
        assert c.cfg.stt_live_floor == 5, c.cfg.stt_live_floor
    print("  [ok] stt_live_floor flows policy -> TieredConfig (default 1)")


def test_empty_slow_tier_counts_unused_not_starved():
    """Short documents never populate STT (VRAM never overflows, so nothing
    demotes): the exposure veto then hides nothing, and must not cry wolf.
    Starvation is reserved for hidden live content; empty tiers count as
    unused, silently. (This is where the old '7 starved steps on hotpotqa'
    came from -- short samples' whole decodes, all harmless.)"""
    import io, contextlib
    p, past = _tiered(seq=32, vram=64, stt=128)
    p(past)                                    # bifurcate: everything fits VRAM
    for c in p.layer_caches.values():
        assert len(c.stt_pos) == 0, "short prompt should leave STT empty"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p._visible_tuple(past)
    assert "WARNING" not in buf.getvalue(), "empty tier should not warn"
    s = p.stats()
    assert s["stt_starved_steps"] == 0, s["stt_starved_steps"]
    # 2: bifurcate's own _visible_tuple call plus the explicit one above --
    # both see an empty slow tier on a short prompt.
    assert s["stt_unused_steps"] == 2, s["stt_unused_steps"]
    print("  [ok] empty slow tier counts as unused (short-doc), not starved")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nAll longbench policy tests passed.")
