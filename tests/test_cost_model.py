"""
test_cost_model.py
------------------
Unit conversions and event pricing for the analytical tier model.

The latency/energy figures in the paper are derived, not measured, so an
arithmetic slip here propagates straight into a headline number with nothing
to catch it. These tests pin the conversions against hand-computed values.

They deliberately do NOT assert that the STT-RAM constants are physically
right -- they are hand-chosen and cost_model.py says so. They assert that
whatever constants are configured are applied consistently.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cost_model import CostModel, DEFAULT_TIERS


def test_read_latency_matches_bandwidth_by_hand():
    cm = CostModel(bytes_per_elem=2)
    elems = 500_000_000          # 1 GB at 2 bytes/elem
    # 1 GB at 864 GB/s = 1/864 s = 1157.4 us
    lat = cm.read_latency_us("GPU_VRAM", elems)
    assert abs(lat - 1e6 / 864.0) < 1e-3, lat
    print(f"  [ok] 1 GB VRAM read = {lat:.1f} us at 864 GB/s")


def test_energy_matches_pj_per_bit_by_hand():
    cm = CostModel(bytes_per_elem=2)
    elems = 1000                 # 2000 bytes = 16000 bits
    # 16000 bits * 12 pJ/bit = 192000 pJ = 192 nJ
    eng = cm.read_energy_nj("GPU_VRAM", elems)
    assert abs(eng - 192.0) < 1e-6, eng
    print(f"  [ok] 1000 elems VRAM read = {eng:.1f} nJ at 12 pJ/bit")


def test_bytes_per_elem_scales_both_latency_and_energy():
    fp16, fp32 = CostModel(bytes_per_elem=2), CostModel(bytes_per_elem=4)
    assert abs(fp32.read_latency_us("STT-RAM", 1000)
               - 2 * fp16.read_latency_us("STT-RAM", 1000)) < 1e-9
    assert abs(fp32.read_energy_nj("STT-RAM", 1000)
               - 2 * fp16.read_energy_nj("STT-RAM", 1000)) < 1e-9
    print("  [ok] doubling bytes_per_elem doubles latency and energy")


def test_write_is_more_expensive_than_read_in_stt_ram():
    """The asymmetry the whole inclusive-shadow design exists to exploit."""
    cm = CostModel(bytes_per_elem=2)
    n = 100_000
    assert cm.write_latency_us("STT-RAM", n) > cm.read_latency_us("STT-RAM", n)
    assert cm.write_energy_nj("STT-RAM", n) > cm.read_energy_nj("STT-RAM", n)
    ratio_lat = cm.write_latency_us("STT-RAM", n) / cm.read_latency_us("STT-RAM", n)
    ratio_eng = cm.write_energy_nj("STT-RAM", n) / cm.read_energy_nj("STT-RAM", n)
    assert abs(ratio_lat - 4.0) < 1e-6, ratio_lat
    assert abs(ratio_eng - 4.0) < 1e-6, ratio_eng
    print(f"  [ok] STT-RAM write/read asymmetry is {ratio_lat:.0f}x latency, "
          f"{ratio_eng:.0f}x energy")


def test_drop_cost_bills_only_the_retiring_read():
    """A dropped token is freed, not migrated: its bill is one STT read."""
    cm = CostModel(bytes_per_elem=2)
    n = 100_000 * 2 * 8 * 128
    d_lat, d_eng = cm.drop_cost(n)
    # drop_cost must be exactly the STT-RAM read, nothing else
    assert abs(d_lat - cm.read_latency_us("STT-RAM", n)) < 1e-9
    assert abs(d_eng - cm.read_energy_nj("STT-RAM", n)) < 1e-9
    print(f"  [ok] drop_cost {d_lat:.1f} us is exactly one STT-RAM read")


def test_promote_and_demote_price_the_right_directions():
    cm = CostModel(bytes_per_elem=2)
    n = 10_000
    # promote = STT read + VRAM write; demote = VRAM read + STT write
    p_lat, _ = cm.promote_cost(n)
    d_lat, _ = cm.demote_cost(n)
    assert abs(p_lat - (cm.read_latency_us("STT-RAM", n)
                        + cm.write_latency_us("GPU_VRAM", n))) < 1e-9
    assert abs(d_lat - (cm.read_latency_us("GPU_VRAM", n)
                        + cm.write_latency_us("STT-RAM", n))) < 1e-9
    # demotion is the expensive direction -- that is the design's premise
    assert d_lat > p_lat, "demote should cost more than promote (STT write penalty)"
    print(f"  [ok] demote ({d_lat:.3f} us) costs more than promote "
          f"({p_lat:.3f} us), as the write penalty requires")


def test_leakage_is_computed_and_favours_stt_ram():
    cm = CostModel(bytes_per_elem=2)
    elems = 1024 * 2 * 8 * 128 * 32          # 1024 tokens, 32 layers
    vram = cm.leakage_energy_nj("GPU_VRAM", elems, 1.0)
    stt = cm.leakage_energy_nj("STT-RAM", elems, 1.0)
    assert stt < vram, (stt, vram)
    # ratio must track the configured mW/MB exactly
    assert abs(vram / stt - DEFAULT_TIERS["GPU_VRAM"].leakage_mw_per_mb
               / DEFAULT_TIERS["STT-RAM"].leakage_mw_per_mb) < 1e-6
    assert cm.leakage_energy_nj("STT-RAM", elems, 0.0) == 0.0
    print(f"  [ok] leakage over 1 s: STT {stt/1e6:.1f} mJ < VRAM {vram/1e6:.1f} mJ")


def test_baseline_attention_cost_scales_with_attended_tokens():
    cm = CostModel(bytes_per_elem=2)
    a, _ = cm.baseline_attention_cost(1024, 8, 128)
    b, _ = cm.baseline_attention_cost(2048, 8, 128)
    assert abs(b - 2 * a) < 1e-9, (a, b)
    print("  [ok] baseline attention cost is linear in attended tokens")


def test_slow_tier_is_slower_than_fast_tier():
    """Regression: STT-RAM was once priced at 1 TB/s read vs VRAM's 864 GB/s,
    i.e. the nominally slow victim tier read FASTER and cheaper than the fast
    tier — under which no promotion policy can be justified. Cited anchors
    (Everspin ~2.66 GB/s/chip, Li et al. 2024 prototype 26.7 GB/s) put any
    honest STT figure far below GDDR6."""
    assert DEFAULT_TIERS["STT-RAM"].read_bw_gbps < DEFAULT_TIERS["GPU_VRAM"].read_bw_gbps, (
        DEFAULT_TIERS["STT-RAM"].read_bw_gbps, DEFAULT_TIERS["GPU_VRAM"].read_bw_gbps)
    assert DEFAULT_TIERS["STT-RAM"].write_bw_gbps <= DEFAULT_TIERS["STT-RAM"].read_bw_gbps
    print(f"  [ok] STT read {DEFAULT_TIERS['STT-RAM'].read_bw_gbps} GB/s < "
          f"VRAM {DEFAULT_TIERS['GPU_VRAM'].read_bw_gbps} GB/s")


if __name__ == "__main__":
    test_read_latency_matches_bandwidth_by_hand()
    test_energy_matches_pj_per_bit_by_hand()
    test_bytes_per_elem_scales_both_latency_and_energy()
    test_write_is_more_expensive_than_read_in_stt_ram()
    test_drop_cost_bills_only_the_retiring_read()
    test_promote_and_demote_price_the_right_directions()
    test_leakage_is_computed_and_favours_stt_ram()
    test_baseline_attention_cost_scales_with_attended_tokens()
    print("\nAll cost model tests passed.")
