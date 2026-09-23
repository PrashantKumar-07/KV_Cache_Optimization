# baselines.py — reference cache policies for head-to-head comparison
#
# All baselines share the same decode interface:
#   reset_prompt(k, v, q)  -> occupancy dict
#   step(q, new_k, new_v, pos) -> attn output (H, 1, D)
#
# Accuracy gaps vs FullAttention isolate the effect of each eviction policy.
#
# SCOPE — READ THIS BEFORE CITING ANY NUMBER FROM THIS FILE.
# These standalone CPU-simulator baselines serve the unit tests and direct
# cache drivers only. They did NOT produce the LongBench results: published
# baseline scores come from the separate H2O/SnapKV/StreamingLLM policies in
# experiments/longbench_eval.py, which run on a real model's DynamicCache
# instead of (H, seq, D) tensors. Same budget/sink semantics in both (pinned
# by tests/test_baselines.py), different interfaces.

from __future__ import annotations
import torch
from attention import scaled_dot_product_attention, snapkv_importance


class FullAttention:
    """Oracle — keeps every token. Upper bound on accuracy."""

    def __init__(self, device="cpu", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.k = self.v = None
        self.total_macs = 0

    def reset_prompt(self, k_all, v_all, q_all=None):
        self.k = k_all.clone()
        self.v = v_all.clone()
        return {"kept": k_all.shape[1]}

    def step(self, q, new_k, new_v, new_pos):
        self.k = torch.cat([self.k, new_k], dim=1)
        self.v = torch.cat([self.v, new_v], dim=1)
        out, _, macs = scaled_dot_product_attention(q, self.k, self.v)
        self.total_macs += macs
        return out

    @property
    def num_tokens(self):
        return self.k.shape[1]


class StreamingLLM:
    """Sinks + sliding window; everything else permanently dropped.

    Ref: Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024.
    """

    def __init__(self, sink_size=4, window_size=64, device="cpu", dtype=torch.float32):
        self.sink_size = sink_size
        self.window_size = window_size
        self.device, self.dtype = device, dtype
        self.k = self.v = None
        self.pos = []
        self.total_macs = 0

    def reset_prompt(self, k_all, v_all, q_all=None):
        N = k_all.shape[1]
        sink = list(range(min(self.sink_size, N)))
        win = list(range(max(0, N - self.window_size), N))
        idx = sorted(set(sink) | set(win))
        self.k = k_all[:, idx, :].clone()
        self.v = v_all[:, idx, :].clone()
        self.pos = list(idx)
        return {"kept": len(idx)}

    def step(self, q, new_k, new_v, new_pos):
        self.k = torch.cat([self.k, new_k], dim=1)
        self.v = torch.cat([self.v, new_v], dim=1)
        self.pos.append(new_pos)
        out, _, macs = scaled_dot_product_attention(q, self.k, self.v)
        self.total_macs += macs

        # trim to sinks + last window_size tokens
        max_pos = max(self.pos)
        keep = [i for i, p in enumerate(self.pos)
                if p < self.sink_size or p > max_pos - self.window_size]
        self.k = self.k[:, keep, :]
        self.v = self.v[:, keep, :]
        self.pos = [self.pos[i] for i in keep]
        return out

    @property
    def num_tokens(self):
        return self.k.shape[1]


class H2O:
    """Heavy-Hitter Oracle — evicts lowest cumulative-attention token each step.

    Keeps sinks + recent window pinned; the remainder competes by accumulated
    attention weight. The loser of each step is permanently dropped.

    Ref: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference", NeurIPS 2023.
    """

    def __init__(self, budget=64, sink_size=4, window_size=16, device="cpu", dtype=torch.float32):
        self.budget = budget
        self.sink_size = sink_size
        self.window_size = window_size
        self.device, self.dtype = device, dtype
        self.k = self.v = None
        self.pos = []
        self.cum = None
        self.total_macs = 0

    def reset_prompt(self, k_all, v_all, q_all=None):
        """Seed the cache with `budget` prompt tokens, not just sink+window.

        This used to keep only sinks + the recent window, discarding the rest
        of the budget outright: at budget=1024 on a 4000-token prompt it kept
        20 tokens and threw away 1004 tokens' worth of allocation, which
        crippled the baseline TieredKV is measured against. H2O has no
        accumulated attention yet at prompt time, so the remaining budget is
        filled by observation-window importance -- the same signal
        H2OPolicy._initialize_scores uses in the LongBench harness, so the
        simulator and the harness agree on what H2O starts from.
        """
        N = k_all.shape[1]
        w = min(self.window_size, N)
        sink = set(range(min(self.sink_size, N)))
        win = set(range(max(0, N - w), N))
        pinned = sink | win

        # K-as-Q proxy when no real queries are supplied, matching
        # H2OPolicy._initialize_scores in experiments/longbench_eval.py.
        q_win = (q_all if q_all is not None else k_all)[:, N - w:, :]
        imp = snapkv_importance(q_win, k_all, pool_kernel=1)

        extra = max(0, self.budget - len(pinned))
        rest = sorted((i for i in range(N) if i not in pinned),
                      key=lambda i: imp[i].item(), reverse=True)
        idx = sorted(pinned | set(rest[:extra]))

        self.k = k_all[:, idx, :].clone()
        self.v = v_all[:, idx, :].clone()
        self.pos = list(idx)
        self.cum = imp[idx].clone().to(self.dtype)
        return {"kept": len(idx)}

    def step(self, q, new_k, new_v, new_pos):
        self.k = torch.cat([self.k, new_k], dim=1)
        self.v = torch.cat([self.v, new_v], dim=1)
        self.pos.append(new_pos)
        self.cum = torch.cat([self.cum, torch.zeros(1, device=self.device, dtype=self.dtype)])

        out, attn_w, macs = scaled_dot_product_attention(q, self.k, self.v)
        self.total_macs += macs
        self.cum += attn_w.squeeze(1).sum(0)  # accumulate attention mass per token

        # evict lowest-scoring unprotected token until within budget
        while len(self.pos) > self.budget:
            max_pos = max(self.pos)
            cand = self.cum.clone()
            for i, p in enumerate(self.pos):
                if p < self.sink_size or p > max_pos - self.window_size:
                    cand[i] = float("inf")
            # Every remaining token is sink- or window-protected (reachable
            # whenever budget < sink_size + window_size). argmin over an
            # all-inf vector returns index 0, which is the FIRST ATTENTION
            # SINK -- measured to wipe positions 0-3 within three steps at
            # budget=16, sink=4, window=16. Capacity cannot be met without
            # evicting a protected token, so stop instead: overshooting the
            # budget is recoverable, destroying the sinks is not.
            if not torch.isfinite(cand).any():
                break
            victim = int(torch.argmin(cand).item())
            keep = [i for i in range(len(self.pos)) if i != victim]
            self.k = self.k[:, keep, :]
            self.v = self.v[:, keep, :]
            self.pos = [self.pos[i] for i in keep]
            self.cum = self.cum[keep]
        return out

    @property
    def num_tokens(self):
        return self.k.shape[1]


class SnapKV:
    """SnapKV prompt compression + H2O-style decode eviction.

    Prompt is compressed using SnapKV importance (pooled attention from the
    observation window). Decode eviction follows H2O (cumulative attention).
    Budget is held fixed throughout; eviction is permanent.

    Ref: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation", NeurIPS 2024.
    """

    def __init__(self, budget=64, sink_size=4, window_size=16,
                 pool_kernel=5, device="cpu", dtype=torch.float32):
        self.budget = budget
        self.sink_size = sink_size
        self.window_size = window_size
        self.pool_kernel = pool_kernel
        self.device, self.dtype = device, dtype
        self.k = self.v = None
        self.pos = []
        self.cum = None
        self.total_macs = 0

    def reset_prompt(self, k_all, v_all, q_all):
        N = k_all.shape[1]
        w = min(self.window_size, N)
        imp = snapkv_importance(q_all[:, N - w:, :], k_all, self.pool_kernel)

        sink = list(range(min(self.sink_size, N)))
        win = list(range(max(0, N - w), N))
        pinned = set(sink) | set(win)
        rest = sorted([i for i in range(N) if i not in pinned],
                      key=lambda i: imp[i].item(), reverse=True)
        extra = max(0, self.budget - len(pinned))
        idx = sorted(pinned | set(rest[:extra]))

        self.k = k_all[:, idx, :].clone()
        self.v = v_all[:, idx, :].clone()
        self.pos = list(idx)
        self.cum = imp[idx].clone().to(self.dtype)
        return {"kept": len(idx)}

    def step(self, q, new_k, new_v, new_pos):
        self.k = torch.cat([self.k, new_k], dim=1)
        self.v = torch.cat([self.v, new_v], dim=1)
        self.pos.append(new_pos)
        self.cum = torch.cat([self.cum, torch.zeros(1, device=self.device, dtype=self.dtype)])

        out, attn_w, macs = scaled_dot_product_attention(q, self.k, self.v)
        self.total_macs += macs
        self.cum += attn_w.squeeze(1).sum(0)

        while len(self.pos) > self.budget:
            max_pos = max(self.pos)
            cand = self.cum.clone()
            for i, p in enumerate(self.pos):
                if p < self.sink_size or p > max_pos - self.window_size:
                    cand[i] = float("inf")
            # See H2O.step: an all-inf candidate vector means every survivor
            # is sink/window protected, and argmin would return index 0 --
            # the first attention sink. Stop rather than evict it.
            if not torch.isfinite(cand).any():
                break
            victim = int(torch.argmin(cand).item())
            keep = [i for i in range(len(self.pos)) if i != victim]
            self.k = self.k[:, keep, :]
            self.v = self.v[:, keep, :]
            self.pos = [self.pos[i] for i in keep]
            self.cum = self.cum[keep]
        return out

    @property
    def num_tokens(self):
        return self.k.shape[1]


class Quest:
    """Quest — query-aware sparse attention via page-level min/max sketching.

    Keeps the FULL KV cache resident (same memory as FullAttention). Each
    decode step scores pages with a cheap upper-bound sketch and attends only
    the top-k pages, reducing per-step MACs without evicting anything.

    Ref: Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024.
    """

    def __init__(self, page_size=16, attend_top_pages=4,
                 sink_size=4, window_size=16, device="cpu", dtype=torch.float32):
        self.page_size = page_size
        self.attend_top_pages = attend_top_pages
        self.sink_size = sink_size
        self.window_size = window_size
        self.device, self.dtype = device, dtype
        self.k = self.v = None
        self.pos = []
        self.total_macs = 0
        self.total_sketch_macs = 0

    def reset_prompt(self, k_all, v_all, q_all=None):
        self.k = k_all.clone()
        self.v = v_all.clone()
        self.pos = list(range(k_all.shape[1]))
        return {"kept": k_all.shape[1]}

    def _sketch_score(self, q):
        """Per-page upper bound: sum_d max(q_d*min_d, q_d*max_d), summed over heads."""
        H, N, D = self.k.shape
        P = self.page_size
        num_pages = (N + P - 1) // P
        scores = torch.zeros(num_pages, device=self.device, dtype=self.dtype)

        for pi in range(num_pages):
            s, e = pi * P, min((pi + 1) * P, N)
            page_k = self.k[:, s:e, :]
            min_k = page_k.min(dim=1).values
            max_k = page_k.max(dim=1).values
            q_sq = q.squeeze(1)
            # Per-FEATURE-dimension max, then reduce over D (Quest, Sec. 3.2).
            # Reducing over D first and taking the max afterwards is not an
            # upper bound: it failed on 60.9% of random mixed-sign queries.
            # See the matching note in TieredKVCache.sketch_check().
            scores[pi] = torch.maximum(q_sq * min_k, q_sq * max_k).sum(-1).sum()

        self.total_sketch_macs += 2 * H * D * num_pages
        return scores, num_pages

    def step(self, q, new_k, new_v, new_pos):
        self.k = torch.cat([self.k, new_k], dim=1)
        self.v = torch.cat([self.v, new_v], dim=1)
        self.pos.append(new_pos)

        N = self.k.shape[1]
        sink_idx = [i for i, p in enumerate(self.pos) if p < self.sink_size]
        win_idx = [i for i, p in enumerate(self.pos)
                   if p >= max(self.pos) - self.window_size + 1]
        pinned = set(sink_idx) | set(win_idx)

        page_scores, num_pages = self._sketch_score(q)
        page_order = torch.argsort(page_scores, descending=True).tolist()

        selected = list(pinned)
        pages_used = 0
        for pi in page_order:
            if pages_used >= self.attend_top_pages:
                break
            s, e = pi * self.page_size, min((pi + 1) * self.page_size, N)
            new_toks = [t for t in range(s, e) if t not in pinned]
            if new_toks:
                selected.extend(new_toks)
                pages_used += 1

        attend_idx = sorted(set(selected))
        out, _, macs = scaled_dot_product_attention(
            q, self.k[:, attend_idx, :], self.v[:, attend_idx, :]
        )
        self.total_macs += macs
        return out

    @property
    def num_tokens(self):
        return self.k.shape[1]
