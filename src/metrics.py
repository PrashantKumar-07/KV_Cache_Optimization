"""
metrics.py
----------
Per-step and aggregate metrics collection.  Everything a reviewer asks for is
recorded here so experiments never rely on "assumptions":

  - MACs / GOPs            (compute)
  - bytes moved per tier   (traffic)
  - migration counts       (promotions / demotions / drops)
  - tier occupancy         (memory footprint over time)
  - derived latency/energy (via cost_model)
  - accuracy vs oracle     (filled in by the experiment, not the cache)

A StepRecord is emitted every decode step; RunMetrics aggregates a whole run.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List
import json


@dataclass
class StepRecord:
    step: int = 0
    # compute
    attn_macs: int = 0
    sketch_macs: int = 0
    # migration (element counts, per KV so multiply by 2 for K+V where noted)
    promoted_tokens: int = 0
    demoted_tokens: int = 0
    dropped_tokens: int = 0
    # occupancy (token counts)
    sram_tokens: int = 0
    sttram_tokens: int = 0
    dram_tokens: int = 0
    # derived (filled by cost model)
    latency_us: float = 0.0
    energy_nj: float = 0.0
    # sub-latency breakdown
    lat_sketch_us: float = 0.0
    lat_promote_us: float = 0.0
    lat_attention_us: float = 0.0
    lat_demote_us: float = 0.0


@dataclass
class RunMetrics:
    label: str = ""
    steps: List[StepRecord] = field(default_factory=list)
    # config echo
    num_heads: int = 0
    head_dim: int = 0
    prompt_len: int = 0

    def add(self, rec: StepRecord):
        self.steps.append(rec)

    # ---- aggregates ----------------------------------------------------------
    @property
    def total_macs(self) -> int:
        return sum(s.attn_macs + s.sketch_macs for s in self.steps)

    @property
    def total_gops(self) -> float:
        # 1 MAC = 2 ops (a multiply + an add)
        return self.total_macs * 2 / 1e9

    @property
    def total_promoted(self) -> int:
        return sum(s.promoted_tokens for s in self.steps)

    @property
    def total_demoted(self) -> int:
        return sum(s.demoted_tokens for s in self.steps)

    @property
    def total_dropped(self) -> int:
        return sum(s.dropped_tokens for s in self.steps)

    @property
    def total_latency_us(self) -> float:
        return sum(s.latency_us for s in self.steps)

    @property
    def total_energy_nj(self) -> float:
        return sum(s.energy_nj for s in self.steps)

    @property
    def peak_sram_tokens(self) -> int:
        return max((s.sram_tokens for s in self.steps), default=0)

    @property
    def peak_sttram_tokens(self) -> int:
        return max((s.sttram_tokens for s in self.steps), default=0)

    def kv_bytes(self, tokens: int, bytes_per_elem: int = 2) -> int:
        """Memory footprint in bytes for `tokens` KV entries (K and V)."""
        return tokens * 2 * self.num_heads * self.head_dim * bytes_per_elem

    def summary(self) -> dict:
        return {
            "label": self.label,
            "steps": len(self.steps),
            "total_GOPs": round(self.total_gops, 4),
            "total_MACs": self.total_macs,
            "total_latency_us": round(self.total_latency_us, 3),
            "total_energy_nj": round(self.total_energy_nj, 3),
            "promoted": self.total_promoted,
            "demoted": self.total_demoted,
            "dropped": self.total_dropped,
            "peak_sram_tokens": self.peak_sram_tokens,
            "peak_sttram_tokens": self.peak_sttram_tokens,
            "peak_sram_KB": round(self.kv_bytes(self.peak_sram_tokens) / 1024, 2),
            "peak_sttram_KB": round(self.kv_bytes(self.peak_sttram_tokens) / 1024, 2),
        }

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"summary": self.summary(),
                 "config": {"num_heads": self.num_heads,
                            "head_dim": self.head_dim,
                            "prompt_len": self.prompt_len},
                 "steps": [asdict(s) for s in self.steps]},
                f, indent=2,
            )
