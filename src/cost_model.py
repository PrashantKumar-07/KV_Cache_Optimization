# cost_model.py — analytical latency + energy for the 2-tier KV cache hierarchy
#
# STT-RAM is simulated, not physical. Every byte moved between the two tiers
# is costed with per-bit bandwidth/energy figures.
#
# GPU VRAM bandwidth is the L40S GDDR6 specification (864 GB/s); its
# energy/leakage prices below are assumed. The STT tier has no physical device
# in this prototype. Its 21/5.25, 50/12.5, and 100/25 GB/s pairs are
# hypothetical sensitivity inputs, including an unvalidated 4x read/write
# bandwidth ratio. They are not measurements of three deployable systems.
# On the evaluated L40S PCIe Gen4x16 link, even the nominal 50 GB/s read
# input exceeds the ~32 GB/s one-way link ceiling before protocol overhead;
# it therefore cannot be interpreted as GPU-visible device bandwidth.
# Everspin reports 2.7 GB/s for both reads and writes per 1Gb device; even
# eight-device striping would require a controller and GPU-accessible link
# before its aggregate could be used here. Li et al.'s 26.7 GB/s result is an
# array-level read for a different design, while Melody measures CXL memory,
# not STT-MRAM. The 2/8 pJ/bit prices are also assumptions: decode KV-traffic
# accounting excludes device interface, compute, prompt placement, and system
# power. See README.md (Hardware & Cost Model) for sources and limitations.
#
# Model scores and GPU timing are measured; STT traffic/energy is derived.
# Treat the device-cost outputs as illustrative until physically validated.

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
    leakage_mw_per_mb: float # mW per MiB held; legacy field name retained


# ---- Assumed cost tiers, with L40S-specified VRAM bandwidth ---------------
# Tier 1: GPU VRAM — L40S uses GDDR6, 864 GB/s; 12 pJ/bit assumed.
#         This is the hot working set during GPU inference.
#         (NOT on-chip SRAM; that would be 19.5 TB/s, 1 pJ/bit — wrong tier.)
# Tier 2: STT-RAM — simulated; 4x read/write asymmetry is an assumption.
DEFAULT_TIERS = {
    "GPU_VRAM": TierSpec(
        name="GPU_VRAM",
        read_bw_gbps=864.0,   write_bw_gbps=864.0,    # L40S GDDR6, 384-bit @ 18 Gbps
        read_pj_per_bit=12.0, write_pj_per_bit=12.0,  # GDDR6 energy (literature)
        leakage_mw_per_mb=8.0,                         # GDDR6 DRAM standby leakage
    ),
    "STT-RAM": TierSpec(
        name="STT-RAM",
        read_bw_gbps=50.0,   write_bw_gbps=12.5,    # assumed 4x asymmetry
        read_pj_per_bit=2.0,  write_pj_per_bit=8.0, # assumed array costs
        leakage_mw_per_mb=1.0,                     # assumed standby price
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
    bytes_per_elem: int = 2          # default KV storage width
    tier_bytes_per_elem: dict = field(default_factory=dict)

    def storage_bytes(self, tier, num_elems):
        return num_elems * self.tier_bytes_per_elem.get(tier, self.bytes_per_elem)

    # Bandwidth scenarios (read, write) in GB/s. The commercial anchor is
    # ideal interface arithmetic for two 1Gb devices, not a measured
    # GPU-accessible implementation. All energy prices remain assumptions.
    STT_BW_TENTPOLES = {
        "commercial_2x1gb_ideal": (5.4, 5.4),
        "pessimistic": (21.0, 5.25),
        "nominal": (50.0, 12.5),
        "optimistic": (100.0, 25.0),
    }

    @classmethod
    def with_stt_bw(cls, label_or_read, write_bw_gbps=None, bytes_per_elem=2):
        """Build a CostModel with an assumed STT-RAM bandwidth scenario.

        Pass a scenario label or an explicit (read, write) pair. The
        commercial_2x1gb_ideal label uses perfect striping of Everspin's
        per-device interface claim; controller/link losses are excluded.
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
        gb = self.storage_bytes(tier, num_elems) / 1e9
        return gb / self.tiers[tier].read_bw_gbps * 1e6      # -> microseconds

    def write_latency_us(self, tier: str, num_elems: int) -> float:
        gb = self.storage_bytes(tier, num_elems) / 1e9
        return gb / self.tiers[tier].write_bw_gbps * 1e6

    def read_energy_nj(self, tier: str, num_elems: int) -> float:
        return self.storage_bytes(tier, num_elems) * 8 * self.tiers[tier].read_pj_per_bit / 1e3   # pJ->nJ

    def write_energy_nj(self, tier: str, num_elems: int) -> float:
        return self.storage_bytes(tier, num_elems) * 8 * self.tiers[tier].write_pj_per_bit / 1e3

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
        """Invalidate a cold row without transferring its K/V payload.

        A physical allocator/controller may incur metadata work, but this
        prototype neither performs nor measures a payload read on drop.
        """
        return 0.0, 0.0

    def leakage_energy_nj(self, tier: str, num_elems: int, seconds: float) -> float:
        """Static (standby) energy for holding `num_elems` resident in `tier`.

        leakage_mw_per_mb was declared on TierSpec from the start but never
        read by anything, which meant the energy model captured only dynamic
        read/write traffic. That omission works directly against the paper's
        own argument: near-zero standby leakage is STT-RAM's main physical
        advantage over DRAM and SRAM (1.0 mW/MiB here against GDDR6's 8.0 and
        DDR5's 15.0), so a dynamic-only model silently discards the effect
        the tiered hierarchy is supposed to exploit.

        Reported as a SEPARATE figure rather than folded into energy_nj, so
        that existing dynamic-energy comparisons keep their old meaning and
        the two contributions stay individually auditable.
        """
        mib = self.storage_bytes(tier, num_elems) / (1024 * 1024)
        mw = mib * self.tiers[tier].leakage_mw_per_mb
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

    def sketch_read_cost(self, num_elems: int, tier: str = "STT-RAM", *,
                         metadata_bytes_per_elem: int | None = None):
        """
        Ideal persistent page-bound read cost; excludes construction,
        maintenance, index storage, scoring compute and interface overhead.

        TieredKVCache reconstructs bounds in the emulator but analytically
        assumes persistent metadata in STT-RAM. Its metadata width is passed
        separately from the cold KV payload width.
        Quest: sketches are read over the full cache in GPU VRAM — use tier='GPU_VRAM'.
        Pass tier explicitly; the default STT-RAM is for the TieredKV path.
        """
        if metadata_bytes_per_elem is None:
            return self.read_latency_us(tier, num_elems), self.read_energy_nj(tier, num_elems)
        if metadata_bytes_per_elem not in (1, 2, 4, 8):
            raise ValueError('Invalid sketch metadata element width')
        byte_count = num_elems * metadata_bytes_per_elem
        return (byte_count / 1e9 / self.tiers[tier].read_bw_gbps * 1e6,
                byte_count * 8 * self.tiers[tier].read_pj_per_bit / 1e3)

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
