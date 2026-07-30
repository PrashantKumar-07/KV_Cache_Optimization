"""
tiered_kv_cache.py
------------------
The core contribution: a 3-tier KV-cache memory hierarchy with query-aware
routing.  Pure PyTorch, hardware-agnostic, instrumented end to end.

    Tier 1  SRAM     - hot working set  (sinks + recent window + hot heavy-hitters)
    Tier 2  STT-RAM  - warm victim cache (recently evicted, sketch-indexed)
    Tier 3  DRAM/Drop- cold storage / permanent eviction

Decode-step pipeline (Phase 1), executed every generated token:

    1. sketch_check   - Quest-style min/max upper-bound scores over STT-RAM pages
    2. promote        - fetch predicted-useful pages STT-RAM -> SRAM
    3. compute_attn   - exact attention over the SRAM working set only
    4. evict_demote   - SRAM->STT-RAM (lowest cum-attn), STT-RAM->DRAM (LRU)

The attention MATH is identical to full attention; only the visible K/V set
changes.  Attention sinks (positions 0..sink_size-1) and the recent sliding
window are never evicted.

Tensor convention everywhere:  (num_heads H, seq, head_dim D).
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from attention import scaled_dot_product_attention, snapkv_importance
from cost_model import CostModel
from metrics import StepRecord, RunMetrics


@dataclass
class TieredConfig:
    num_heads: int = 8
    head_dim: int = 64
    sink_size: int = 4           # attention sinks pinned to SRAM forever
    window_size: int = 16        # recent sliding window pinned to SRAM
    sram_capacity: int = 64      # max tokens resident in SRAM
    sttram_capacity: int = 128   # max tokens resident in STT-RAM
    page_size: int = 16          # Quest sketch granularity
    promote_top_pages: int = 2   # max pages promoted per step
    store_dram: bool = False     # True: keep cold tokens in DRAM; False: drop
    pool_kernel: int = 5         # SnapKV clustering pool
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class TieredKVCache:
    def __init__(self, cfg: TieredConfig, cost_model: CostModel | None = None):
        self.cfg = cfg
        self.cm = cost_model or CostModel(bytes_per_elem=2)
        H, D = cfg.num_heads, cfg.head_dim
        dev, dt = cfg.device, cfg.dtype

        # ---- Tier 1: SRAM ----
        self.sram_k = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.sram_v = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.sram_pos = []                                     # original token index
        self.sram_cum = torch.empty(0, device=dev, dtype=dt)   # cumulative attn received

        # ---- Tier 2: STT-RAM ----
        self.stt_k = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.stt_v = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.stt_pos = []
        self.stt_last_access = torch.empty(0, device=dev, dtype=dt)

        # ---- Tier 3: DRAM (optional) ----
        self.dram_k = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.dram_v = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.dram_pos = []

        self.metrics = RunMetrics(label="TieredKV", num_heads=H, head_dim=D)
        self._t = 0  # logical clock for LRU

    # =====================================================================
    # PHASE 0 : initial bifurcation of the prompt
    # =====================================================================
    def initial_bifurcation(self, k_all, v_all, q_all):
        """
        k_all, v_all, q_all : (H, N, D) for the prompt.
        Routes prompt tokens into the three tiers using SnapKV importance,
        with sinks and the recent window force-pinned to SRAM.
        """
        cfg = self.cfg
        H, N, D = k_all.shape
        self.metrics.prompt_len = N

        w = min(cfg.window_size, N)
        q_window = q_all[:, N - w:, :]                       # observation window
        importance = snapkv_importance(q_window, k_all, cfg.pool_kernel)  # (N,)

        sink_idx = list(range(min(cfg.sink_size, N)))
        window_idx = list(range(max(0, N - w), N))
        pinned = sorted(set(sink_idx) | set(window_idx))

        # rank the remaining tokens by importance
        rest = [i for i in range(N) if i not in set(pinned)]
        rest_sorted = sorted(rest, key=lambda i: importance[i].item(), reverse=True)

        sram_budget = max(0, cfg.sram_capacity - len(pinned))
        sram_extra = rest_sorted[:sram_budget]
        after_sram = rest_sorted[sram_budget:]

        stt_sel = after_sram[:cfg.sttram_capacity]
        cold_sel = after_sram[cfg.sttram_capacity:]

        sram_idx = sorted(set(pinned) | set(sram_extra))
        stt_idx = sorted(stt_sel)
        cold_idx = sorted(cold_sel)

        self.sram_k = k_all[:, sram_idx, :].clone()
        self.sram_v = v_all[:, sram_idx, :].clone()
        self.sram_pos = list(sram_idx)
        self.sram_cum = importance[sram_idx].clone().to(self.cfg.dtype)

        self.stt_k = k_all[:, stt_idx, :].clone()
        self.stt_v = v_all[:, stt_idx, :].clone()
        self.stt_pos = list(stt_idx)
        self.stt_last_access = torch.zeros(len(stt_idx), device=cfg.device, dtype=cfg.dtype)

        if cfg.store_dram:
            self.dram_k = k_all[:, cold_idx, :].clone()
            self.dram_v = v_all[:, cold_idx, :].clone()
            self.dram_pos = list(cold_idx)

        return {
            "sram": len(sram_idx), "sttram": len(stt_idx),
            "dram_or_dropped": len(cold_idx), "prompt_len": N,
        }

    # =====================================================================
    # STEP 1 : sketch check  (Quest min/max upper bound)
    # =====================================================================
    def _page_bounds(self):
        """Return per-page (min_k, max_k) sketches and the token-index slices."""
        H, m, D = self.stt_k.shape
        ps = self.cfg.page_size
        pages = []
        for start in range(0, m, ps):
            end = min(start + ps, m)
            block = self.stt_k[:, start:end, :]              # (H, p, D)
            pages.append((start, end,
                          block.min(dim=1).values,           # (H, D)
                          block.max(dim=1).values))          # (H, D)
        return pages

    def sketch_check(self, q):
        """
        q : (H, 1, D).  Returns list of (start, end) token slices to promote,
        plus the sketch MAC count.
        """
        H, m, D = self.stt_k.shape
        if m == 0:
            return [], 0

        pages = self._page_bounds()
        q2 = q.squeeze(1)                                    # (H, D)
        ub_scores = []
        for (s, e, mn, mx) in pages:
            # upper bound = sum_h max(q·min, q·max)
            dmin = (q2 * mn).sum(-1)                          # (H,)
            dmax = (q2 * mx).sum(-1)                          # (H,)
            ub = torch.maximum(dmin, dmax).sum().item()
            ub_scores.append((ub, s, e))

        # threshold: current lowest SRAM relevance for THIS query
        if self.sram_k.shape[1] > 0:
            s_sram = torch.matmul(q, self.sram_k.transpose(-2, -1)).squeeze(1)  # (H, n)
            thresh = s_sram.sum(0).min().item()
        else:
            thresh = float("-inf")

        ub_scores.sort(key=lambda x: x[0], reverse=True)
        hits = [(s, e) for (ub, s, e) in ub_scores[:self.cfg.promote_top_pages] if ub > thresh]

        sketch_macs = 2 * H * len(pages) * D                 # min-dot + max-dot per page
        return hits, sketch_macs

    # =====================================================================
    # STEP 2 : promote  (STT-RAM -> SRAM)
    # =====================================================================
    def promote(self, hits):
        if not hits:
            return 0
        H, m, D = self.stt_k.shape
        keep = torch.ones(m, dtype=torch.bool)
        moved = 0
        for (s, e) in hits:
            idx = list(range(s, e))
            self.sram_k = torch.cat([self.sram_k, self.stt_k[:, idx, :]], dim=1)
            self.sram_v = torch.cat([self.sram_v, self.stt_v[:, idx, :]], dim=1)
            self.sram_pos += [self.stt_pos[i] for i in idx]
            self.sram_cum = torch.cat(
                [self.sram_cum, torch.zeros(len(idx), device=self.cfg.device, dtype=self.cfg.dtype)]
            )
            for i in idx:
                keep[i] = False
            moved += len(idx)

        # remove promoted tokens from STT-RAM
        keep_idx = keep.nonzero(as_tuple=True)[0].tolist()
        self.stt_k = self.stt_k[:, keep_idx, :]
        self.stt_v = self.stt_v[:, keep_idx, :]
        self.stt_pos = [self.stt_pos[i] for i in keep_idx]
        self.stt_last_access = self.stt_last_access[keep_idx]
        return moved

    # =====================================================================
    # STEP 3 : compute attention over the SRAM working set
    # =====================================================================
    def compute_attention(self, q):
        out, attn_w, macs = scaled_dot_product_attention(q, self.sram_k, self.sram_v)
        # accumulate attention received by each SRAM token (sum over heads)
        self.sram_cum += attn_w.squeeze(1).sum(0)
        return out, macs

    # =====================================================================
    # STEP 4 : evict + demote
    # =====================================================================
    def _protected_mask(self):
        """True where a SRAM token must NOT be evicted (sink or recent window)."""
        n = len(self.sram_pos)
        mask = torch.zeros(n, dtype=torch.bool)
        cfg = self.cfg
        max_pos = max(self.sram_pos) if self.sram_pos else 0
        for i, p in enumerate(self.sram_pos):
            if p < cfg.sink_size:                       # attention sink
                mask[i] = True
            elif p > max_pos - cfg.window_size:         # recent sliding window
                mask[i] = True
        return mask

    def evict_and_demote(self, new_k, new_v, new_pos):
        cfg = self.cfg
        dropped = 0
        demoted = 0

        # newly generated token always enters SRAM (it is the newest window token)
        self.sram_k = torch.cat([self.sram_k, new_k], dim=1)
        self.sram_v = torch.cat([self.sram_v, new_v], dim=1)
        self.sram_pos.append(new_pos)
        self.sram_cum = torch.cat(
            [self.sram_cum, torch.zeros(1, device=cfg.device, dtype=cfg.dtype)]
        )

        # ---- SRAM overflow -> demote lowest cum-attn unprotected token ----
        while len(self.sram_pos) > cfg.sram_capacity:
            protected = self._protected_mask()
            cand = self.sram_cum.clone()
            cand[protected] = float("inf")               # never pick protected
            victim = int(torch.argmin(cand).item())
            if protected[victim]:                        # all protected: stop
                break

            vk = self.sram_k[:, victim:victim + 1, :]
            vv = self.sram_v[:, victim:victim + 1, :]
            vpos = self.sram_pos[victim]

            # append to STT-RAM
            self.stt_k = torch.cat([self.stt_k, vk], dim=1)
            self.stt_v = torch.cat([self.stt_v, vv], dim=1)
            self.stt_pos.append(vpos)
            self.stt_last_access = torch.cat(
                [self.stt_last_access, torch.tensor([self._t], device=cfg.device, dtype=cfg.dtype)]
            )
            demoted += 1

            # remove victim from SRAM
            keep = [i for i in range(len(self.sram_pos)) if i != victim]
            self.sram_k = self.sram_k[:, keep, :]
            self.sram_v = self.sram_v[:, keep, :]
            self.sram_pos = [self.sram_pos[i] for i in keep]
            self.sram_cum = self.sram_cum[keep]

        # ---- STT-RAM overflow -> deep-demote LRU token to DRAM/drop ----
        while len(self.stt_pos) > cfg.sttram_capacity:
            lru = int(torch.argmin(self.stt_last_access).item())
            if cfg.store_dram:
                self.dram_k = torch.cat([self.dram_k, self.stt_k[:, lru:lru + 1, :]], dim=1)
                self.dram_v = torch.cat([self.dram_v, self.stt_v[:, lru:lru + 1, :]], dim=1)
                self.dram_pos.append(self.stt_pos[lru])
            keep = [i for i in range(len(self.stt_pos)) if i != lru]
            self.stt_k = self.stt_k[:, keep, :]
            self.stt_v = self.stt_v[:, keep, :]
            self.stt_pos = [self.stt_pos[i] for i in keep]
            self.stt_last_access = self.stt_last_access[keep]
            dropped += 1

        return demoted, dropped

    # =====================================================================
    # Full decode step orchestrator
    # =====================================================================
    def step(self, q, new_k, new_v, new_pos):
        self._t += 1
        H, D = self.cfg.num_heads, self.cfg.head_dim
        rec = StepRecord(step=self._t)

        # 1. sketch check
        hits, sketch_macs = self.sketch_check(q)
        rec.sketch_macs = sketch_macs

        # 2. promote
        n_pages_tokens = sum(e - s for (s, e) in hits)
        rec.promoted_tokens = self.promote(hits)

        # 3. attention on SRAM working set
        out, attn_macs = self.compute_attention(q)
        rec.attn_macs = attn_macs

        # 4. evict + demote (also inserts the new token)
        demoted, dropped = self.evict_and_demote(new_k, new_v, new_pos)
        rec.demoted_tokens = demoted
        rec.dropped_tokens = dropped

        # occupancy
        rec.sram_tokens = len(self.sram_pos)
        rec.sttram_tokens = len(self.stt_pos)
        rec.dram_tokens = len(self.dram_pos)

        # ---- derive latency + energy from the cost model ----
        elems_per_tok = H * D
        lat_p, eng_p = self.cm.promote_cost(rec.promoted_tokens * 2 * elems_per_tok)
        lat_d, eng_d = self.cm.demote_cost(rec.demoted_tokens * 2 * elems_per_tok)
        lat_dd, eng_dd = self.cm.deep_demote_cost(rec.dropped_tokens * 2 * elems_per_tok)
        lat_sr, eng_sr = self.cm.sram_compute_read_cost(rec.sram_tokens * 2 * elems_per_tok)
        # sketch traffic: read min/max sketch (2 * n_pages * H * D) from SRAM-resident sketches
        n_pages = math.ceil(max(rec.sttram_tokens, 1) / self.cfg.page_size)
        lat_sk, eng_sk = self.cm.sram_compute_read_cost(2 * n_pages * H * D)

        rec.lat_sketch_us = lat_sk
        rec.lat_promote_us = lat_p
        rec.lat_attention_us = lat_sr
        rec.lat_demote_us = lat_d + lat_dd
        rec.latency_us = lat_sk + lat_p + lat_sr + lat_d + lat_dd
        rec.energy_nj = eng_sk + eng_p + eng_sr + eng_d + eng_dd

        self.metrics.add(rec)
        return out
