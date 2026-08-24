# cost_model.py — analytical latency + energy for the 3-tier KV cache hierarchy
#
# STT-RAM is simulated, not physical. Every byte moved between tiers is costed
# with NVM-literature per-bit figures (same approach as NVSim / CACTI / Destiny).
#
# Hardware mapping (NVIDIA L40S + AMD EPYC 9754):
#   Tier 1  GPU VRAM  — GDDR6, 864 GB/s, 12 pJ/bit  (NOT on-chip SRAM)
#   Tier 2  STT-RAM   — simulated: 1 TB/s read / 250 GB/s write, 2 / 8 pJ/bit
#   Tier 3  CPU DRAM  — DDR5, conservative 200 GB/s, 20 pJ/bit
#
# Accuracy, MACs, and occupancy are MEASURED. Latency/energy are DERIVED.

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TierSpec:
    """Physical characteristics of one memory tier."""
    name: str
    read_bw_gbps: float      # sustained read bandwidth (GB/s)
    write_bw_gbps: float     # sustained write bandwidth (GB/s)
    read_pj_per_bit: float   # dynamic read energy (pJ/bit)
    write_pj_per_bit: float  # dynamic write energy (pJ/bit)
    leakage_mw_per_mb: float # static/leakage power (mW per MB held per second)


# ---- Server-calibrated tiers (NVIDIA L40S, EPYC 9754) ----------------------
# Tier 1: GPU VRAM — L40S uses GDDR6, 864 GB/s, ~12 pJ/bit.
#         This is the hot working set during GPU inference.
#         (NOT on-chip SRAM; that would be 19.5 TB/s, 1 pJ/bit — wrong tier.)
# Tier 2: STT-RAM — simulated; 4× read/write asymmetry exploited by design.
# Tier 3: CPU DRAM — system RAM offload (FlexGen-style); conservative 200 GB/s.
DEFAULT_TIERS = {
    "GPU_VRAM": TierSpec(
        name="GPU_VRAM",
        read_bw_gbps=864.0,   write_bw_gbps=864.0,    # L40S GDDR6, 384-bit @ 18 Gbps
        read_pj_per_bit=12.0, write_pj_per_bit=12.0,  # GDDR6 energy (literature)
        leakage_mw_per_mb=8.0,                         # GDDR6 DRAM standby leakage
    ),
    "STT-RAM": TierSpec(
        name="STT-RAM",
        read_bw_gbps=1000.0,  write_bw_gbps=250.0,    # write ~4× slower than read
        read_pj_per_bit=2.0,  write_pj_per_bit=8.0,   # write ~4× costlier than read
        leakage_mw_per_mb=1.0,                         # near-zero standby leakage
    ),
    "DRAM": TierSpec(
        name="DRAM",
        read_bw_gbps=200.0,   write_bw_gbps=200.0,    # CPU DRAM (DDR5, conservative)
        read_pj_per_bit=20.0, write_pj_per_bit=20.0,
        leakage_mw_per_mb=15.0,
    ),
}

# Kept for reference / ablation studies.
# Use these if you want to compare against an idealised on-chip SRAM hierarchy
# (e.g., to reproduce NVSim-style cache modelling from desktop/ASIC papers).
IDEAL_SRAM_TIERS = {
    "GPU_VRAM": TierSpec(
        name="GPU_VRAM (ideal SRAM, not physical)",
        read_bw_gbps=19500.0, write_bw_gbps=19500.0,
        read_pj_per_bit=1.0,  write_pj_per_bit=1.0,
        leakage_mw_per_mb=80.0,
    ),
    "STT-RAM": DEFAULT_TIERS["STT-RAM"],
    "DRAM":    DEFAULT_TIERS["DRAM"],
}


@dataclass
class CostModel:
    tiers: dict = field(default_factory=lambda: dict(DEFAULT_TIERS))
    bytes_per_elem: int = 2          # fp16 KV cache

    # ---- primitive costs -----------------------------------------------------
    def _bits(self, num_elems: int) -> int:
        return num_elems * self.bytes_per_elem * 8

    def read_latency_us(self, tier: str, num_elems: int) -> float:
        gb = num_elems * self.bytes_per_elem / 1e9
        return gb / self.tiers[tier].read_bw_gbps * 1e6      # -> microseconds

    def write_latency_us(self, tier: str, num_elems: int) -> float:
        gb = num_elems * self.bytes_per_elem / 1e9
        return gb / self.tiers[tier].write_bw_gbps * 1e6

    def read_energy_nj(self, tier: str, num_elems: int) -> float:
        return self._bits(num_elems) * self.tiers[tier].read_pj_per_bit / 1e3   # pJ->nJ

    def write_energy_nj(self, tier: str, num_elems: int) -> float:
        return self._bits(num_elems) * self.tiers[tier].write_pj_per_bit / 1e3

    # ---- migration events (the interesting part) -----------------------------
    def promote_cost(self, num_elems: int):
        """STT-RAM -> GPU_VRAM: read from STT-RAM, write into GPU VRAM."""
        lat = self.read_latency_us("STT-RAM", num_elems) + self.write_latency_us("GPU_VRAM", num_elems)
        eng = self.read_energy_nj("STT-RAM", num_elems) + self.write_energy_nj("GPU_VRAM", num_elems)
        return lat, eng

    def demote_cost(self, num_elems: int):
        """GPU_VRAM -> STT-RAM: read from GPU VRAM, write into STT-RAM (write penalty bites here)."""
        lat = self.read_latency_us("GPU_VRAM", num_elems) + self.write_latency_us("STT-RAM", num_elems)
        eng = self.read_energy_nj("GPU_VRAM", num_elems) + self.write_energy_nj("STT-RAM", num_elems)
        return lat, eng

    def deep_demote_cost(self, num_elems: int):
        """STT-RAM -> DRAM: read from STT-RAM, write into CPU DRAM."""
        lat = self.read_latency_us("STT-RAM", num_elems) + self.write_latency_us("DRAM", num_elems)
        eng = self.read_energy_nj("STT-RAM", num_elems) + self.write_energy_nj("DRAM", num_elems)
        return lat, eng

    def gpu_vram_read_cost(self, num_elems: int):
        """
        Reading K/V out of GPU VRAM (Tier 1) to feed the attention matmul.

        On the L40S server this is GDDR6 at 864 GB/s, 12 pJ/bit — NOT
        on-chip SRAM.  All baselines (StreamingLLM, SnapKV, Quest) that keep
        their active cache in VRAM are charged this cost per step.
        """
        return self.read_latency_us("GPU_VRAM", num_elems), self.read_energy_nj("GPU_VRAM", num_elems)

    # Backward-compatible alias — older call sites used sram_compute_read_cost.
    def sram_compute_read_cost(self, num_elems: int):
        """Deprecated alias for gpu_vram_read_cost(). Use gpu_vram_read_cost() instead."""
        return self.gpu_vram_read_cost(num_elems)

    def sketch_read_cost(self, num_elems: int, tier: str = "STT-RAM"):
        """
        Cost of reading K/V for page sketch scoring.

        TieredKVCache: sketches are held in STT-RAM (Tier 2) — use tier='STT-RAM'.
        Quest: sketches are read over the full cache in GPU VRAM — use tier='GPU_VRAM'.
        Pass tier explicitly; the default STT-RAM is for the TieredKV path.
        """
        return self.read_latency_us(tier, num_elems), self.read_energy_nj(tier, num_elems)

    def baseline_attention_cost(self, num_tokens: int, num_heads: int, head_dim: int):
        """
        Equitable analytical overhead for permanent-eviction baselines
        (StreamingLLM, SnapKV) per decode step.

        These methods keep their cache in GPU VRAM and read it every step for
        the attention matmul.  Charging them GDDR6 (864 GB/s, 12 pJ/bit) —
        not zero — makes the latency comparison fair.
        """
        # K + V read: tokens × head_dim × num_heads × 2 (K and V) elements
        elems = num_tokens * head_dim * num_heads * 2
        return self.gpu_vram_read_cost(elems)
