"""
cost_model.py
-------------
Analytical latency + energy model for a 3-tier KV-cache memory hierarchy.

Why analytical?  A laptop cannot physically contain STT-RAM, and even on the
server STT-RAM is not a real device you can time.  Architecture papers
(e.g. NVSim, CACTI, Destiny) therefore report latency/energy from *calibrated
per-access and per-bit models*.  We do the same: every byte moved between
tiers is costed with published figures, so the numbers are defensible to a
reviewer and independent of the host machine's actual RAM.

Correctness, accuracy, MACs and tier occupancy are MEASURED empirically.
Latency and energy are DERIVED from this model applied to those measured
byte-movement counts.  We never conflate the two.

All default constants are cited in COST_MODEL_REFERENCES (see README). They are
parameters -- sweep them in experiments; they are not hard truths.
"""

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


# ---- Default calibrated tiers -------------------------------------------------
# Figures are representative order-of-magnitude values from the NVM literature.
# SRAM: fast, symmetric, leaky.  STT-RAM: fast read / slow+costly write, ~zero
# leakage, high density.  DRAM: slow, refresh leakage, cheap per bit to build.
DEFAULT_TIERS = {
    "SRAM": TierSpec(
        name="SRAM",
        read_bw_gbps=19500.0, write_bw_gbps=19500.0,
        read_pj_per_bit=1.0,  write_pj_per_bit=1.0,
        leakage_mw_per_mb=80.0,
    ),
    "STT-RAM": TierSpec(
        name="STT-RAM",
        read_bw_gbps=1000.0,  write_bw_gbps=250.0,     # write ~4x slower than read
        read_pj_per_bit=2.0,  write_pj_per_bit=8.0,    # write ~4x costlier than read
        leakage_mw_per_mb=1.0,                          # near-zero standby leakage
    ),
    "DRAM": TierSpec(
        name="DRAM",
        read_bw_gbps=200.0,   write_bw_gbps=200.0,
        read_pj_per_bit=20.0, write_pj_per_bit=20.0,
        leakage_mw_per_mb=15.0,
    ),
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
        """STT-RAM -> SRAM: read from STT-RAM, write into SRAM."""
        lat = self.read_latency_us("STT-RAM", num_elems) + self.write_latency_us("SRAM", num_elems)
        eng = self.read_energy_nj("STT-RAM", num_elems) + self.write_energy_nj("SRAM", num_elems)
        return lat, eng

    def demote_cost(self, num_elems: int):
        """SRAM -> STT-RAM: read from SRAM, write into STT-RAM (write penalty bites here)."""
        lat = self.read_latency_us("SRAM", num_elems) + self.write_latency_us("STT-RAM", num_elems)
        eng = self.read_energy_nj("SRAM", num_elems) + self.write_energy_nj("STT-RAM", num_elems)
        return lat, eng

    def deep_demote_cost(self, num_elems: int):
        """STT-RAM -> DRAM: read from STT-RAM, write into DRAM."""
        lat = self.read_latency_us("STT-RAM", num_elems) + self.write_latency_us("DRAM", num_elems)
        eng = self.read_energy_nj("STT-RAM", num_elems) + self.write_energy_nj("DRAM", num_elems)
        return lat, eng

    def sram_compute_read_cost(self, num_elems: int):
        """Reading K/V out of SRAM to feed the attention matmul each step."""
        return self.read_latency_us("SRAM", num_elems), self.read_energy_nj("SRAM", num_elems)
