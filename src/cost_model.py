# cost_model.py — analytical latency + energy for the 2-tier KV cache hierarchy
#
# STT-RAM is simulated, not physical. Every byte moved between the two tiers
# is costed with per-bit bandwidth/energy figures.
#
# Hardware mapping (NVIDIA L40S + AMD EPYC 9754):
#   Tier 1  GPU VRAM  — GDDR6, 864 GB/s, 12 pJ/bit  (NOT on-chip SRAM;
#                       these are real, plausible published GDDR6/L40S
#                       figures)
#   Tier 2  STT-RAM   — simulated. Bandwidth follows the NVMExplorer
#                       tentpole methodology (Pentecost et al., HPCA'22):
#                       no single value is physically exact, so the paper
#                       reports a sensitivity sweep over three cited anchors
#                       instead of one hand-picked number --
#                         pessimistic  21 /  5.0 GB/s  shipping DIMM-class:
#                           Everspin EMD4E001G 1Gb ST-DDR4, 1333 MT/s/pin,
#                           x16 = 2.66 GB/s/chip, 8-chip DIMM ~= 21 GB/s
#                           (datasheet EMD4E001GAS2; Xilinx ST-DDR4 app note:
#                           tRCD 135ns vs DDR4-2666 13.5ns, tFAW 240 vs 30ns)
#                         nominal      50 / 12.5 GB/s  prototype/CXL-class:
#                           Li et al. 2024 near-memory prototype 26.7 GB/s
#                           array-level, x2 striped; independently matches a
#                           measured CXL 1.1 x16 type-3 expander at 52 GB/s
#                           (Liu et al., ASPLOS'25 "Melody")
#                         optimistic  100 / 25   GB/s  4-device stripe:
#                           round-robin interleave scales 2.5-3.7x over one
#                           device (Weisgut et al., CXL-Bench 2025)
#                       DEFAULT_TIERS uses nominal. Hard constraint enforced
#                       in tests: BW_STT_read < BW_VRAM (864 GB/s) — a slow
#                       tier reading faster than the fast tier (the withdrawn
#                       1 TB/s value) is indefensible and would make promotion
#                       pointless under the model's own numbers.
#                       Methodology notes, all citable:
#                       Asifuzzaman et al., MEMSYS'17 / TECS'21 (BSC+Everspin):
#                       STT is DDRx-compatible, so non-row timings equal DRAM
#                       and only row ops (tRCD/tRP/tFAW/tRRD) vary — reported
#                       as sensitivity ST-1.2/1.5/2.0x DRAM, with an explicit
#                       warning that NVMain's default STT timings are
#                       unverifiable. Energy 2/8 pJ/bit is array-level
#                       (Zhou et al. ICCAD'09: ~0.2nJ read / ~1.6nJ write per
#                       64B line; NVSim tutorial: 0.007/0.056nJ, ~8x); the 4x
#                       read/write asymmetry sits mid-range of Meza et al.
#                       ISCA'12 (1-8x) and Everspin tFAW (8x). System-level
#                       energy adds DDR IO + controller + CXL/PCIe SerDes
#                       (~5-10 pJ/bit), so these are lower bounds. Endurance
#                       1e10 cycles per the EMD4E001G datasheet (research
#                       projections reach 1e12-1e16); retention 3 months @70C
#                       buffer-grade per the same datasheet.
#
# Accuracy, MACs, and occupancy are MEASURED. Latency/energy are DERIVED
# under the assumed tier model above -- treat as illustrative unless/until
# the STT-RAM figures are replaced with cited values.

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
DEFAULT_TIERS = {
    "GPU_VRAM": TierSpec(
        name="GPU_VRAM",
        read_bw_gbps=864.0,   write_bw_gbps=864.0,    # L40S GDDR6, 384-bit @ 18 Gbps
        read_pj_per_bit=12.0, write_pj_per_bit=12.0,  # GDDR6 energy (literature)
        leakage_mw_per_mb=8.0,                         # GDDR6 DRAM standby leakage
    ),
    "STT-RAM": TierSpec(
        name="STT-RAM",
        read_bw_gbps=50.0,   write_bw_gbps=12.5,    # write ~4× slower than read;
        # nominal tentpole (prototype/CXL-class); see header for the full
        # pessimistic/nominal/optimistic sweep. MUST stay < VRAM 864 GB/s.
        read_pj_per_bit=2.0,  write_pj_per_bit=8.0,   # write ~4× costlier than read
        leakage_mw_per_mb=1.0,                         # near-zero standby leakage
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
}


@dataclass
class CostModel:
    tiers: dict = field(default_factory=lambda: dict(DEFAULT_TIERS))
    bytes_per_elem: int = 2          # fp16 KV cache

    # Cited STT-RAM bandwidth tentpoles (read, write) in GB/s for the paper's
    # sensitivity sweep. See the Tier 2 header comment for sources.
    # Keys are the paper's labels; values keep the 4x read/write asymmetry.
    STT_BW_TENTPOLES = {
        "pessimistic": (21.0, 5.25),    # shipping DIMM-class (Everspin 8-chip)
        "nominal": (50.0, 12.5),        # prototype/CXL-x16-class (default)
        "optimistic": (100.0, 25.0),    # 4-device striped (CXL-Bench scaling)
    }

    @classmethod
    def with_stt_bw(cls, label_or_read, write_bw_gbps=None, bytes_per_elem=2):
        """Build a CostModel with a cited STT-RAM bandwidth point.

        Pass a tentpole label ('pessimistic'/'nominal'/'optimistic') or an
        explicit (read, write) pair. Used by the sensitivity sweep so the
        paper never reports a single hand-picked number.
        """
        if write_bw_gbps is None:
            read, write = cls.STT_BW_TENTPOLES[label_or_read]
        else:
            read, write = label_or_read, write_bw_gbps
        tiers = dict(DEFAULT_TIERS)
        stt = tiers["STT-RAM"]
        tiers["STT-RAM"] = TierSpec(
            name=stt.name, read_bw_gbps=read, write_bw_gbps=write,
            read_pj_per_bit=stt.read_pj_per_bit,
            write_pj_per_bit=stt.write_pj_per_bit,
            leakage_mw_per_mb=stt.leakage_mw_per_mb)
        return cls(tiers=tiers, bytes_per_elem=bytes_per_elem)

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

    def drop_cost(self, num_elems: int):
        """Evict from STT-RAM, freeing the line (dropped tokens go nowhere).

        The only physical work is the STT-RAM read that retires the line.
        """
        return (self.read_latency_us("STT-RAM", num_elems),
                self.read_energy_nj("STT-RAM", num_elems))

    def leakage_energy_nj(self, tier: str, num_elems: int, seconds: float) -> float:
        """Static (standby) energy for holding `num_elems` resident in `tier`.

        leakage_mw_per_mb was declared on TierSpec from the start but never
        read by anything, which meant the energy model captured only dynamic
        read/write traffic. That omission works directly against the paper's
        own argument: near-zero standby leakage is STT-RAM's main physical
        advantage over DRAM and SRAM (1.0 mW/MB here against GDDR6's 8.0 and
        DDR5's 15.0), so a dynamic-only model silently discards the effect
        the tiered hierarchy is supposed to exploit.

        Reported as a SEPARATE figure rather than folded into energy_nj, so
        that existing dynamic-energy comparisons keep their old meaning and
        the two contributions stay individually auditable.
        """
        mb = num_elems * self.bytes_per_elem / (1024 * 1024)
        mw = mb * self.tiers[tier].leakage_mw_per_mb
        return mw * 1e-3 * seconds * 1e9      # W * s = J -> nJ

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
        Equitable analytical overhead for the non-tiered baselines
        (Full/StreamingLLM/H2O/SnapKV) per decode step. Called from
        experiments/longbench_eval.py's generate_with_policy for every
        policy that isn't TieredKV (see _ModeledCostMixin/_BASELINE_COST_MODEL
        there) -- previously unused/dead, so no baseline had a
        modeled_latency_us/modeled_energy_nj figure at all.

        These methods keep their cache in GPU VRAM and read it every step for
        the attention matmul. Charging them GDDR6 (864 GB/s, 12 pJ/bit) —
        not zero — makes the latency comparison against TieredKV fair.
        """
        # K + V read: tokens × head_dim × num_heads × 2 (K and V) elements
        elems = num_tokens * head_dim * num_heads * 2
        return self.gpu_vram_read_cost(elems)
