"""
baselines.py
------------
Reference systems sharing the SAME decode interface as TieredKVCache:

    reset_prompt(k_all, v_all, q_all)   -> occupancy dict
    step(q, new_k, new_v, new_pos)      -> attention output (H,1,D)

  FullAttention  - oracle, keeps every token (upper bound on accuracy)
  StreamingLLM   - sinks + recent window only, permanent drop of the middle
  SnapKV         - prompt compression + fixed budget, PERMANENT eviction

The attention math is identical across all of them (attention.py); the only
difference is which K/V tokens survive.  That isolates the effect of the
memory-management policy -- exactly what a reviewer needs to see.
"""

from __future__ import annotations
import torch
from attention import scaled_dot_product_attention, snapkv_importance


class FullAttention:
    """Oracle: never evicts anything."""
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
    """Sinks + sliding window; everything else permanently dropped."""
    def __init__(self, sink_size=4, window_size=64, device="cpu", dtype=torch.float32):
        self.sink_size, self.window_size = sink_size, window_size
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

        # keep only sinks + last window_size tokens
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


class SnapKV:
    """
    Fixed-budget cache with PERMANENT eviction (the lossy baseline our victim
    cache aims to beat).  Prompt compressed by SnapKV importance; during decode,
    H2O-style cumulative attention decides who gets permanently dropped.
    """
    def __init__(self, budget=64, sink_size=4, window_size=16,
                 pool_kernel=5, device="cpu", dtype=torch.float32):
        self.budget, self.sink_size, self.window_size = budget, sink_size, window_size
        self.pool_kernel = pool_kernel
        self.device, self.dtype = device, dtype
        self.k = self.v = None
        self.pos = []
        self.cum = None
        self.total_macs = 0

    def reset_prompt(self, k_all, v_all, q_all):
        N = k_all.shape[1]
        w = min(self.window_size, N)
        imp = snapkv_importance(q_all[:, N - w:, :], k_all, self.pool_kernel)  # (N,)

        sink = list(range(min(self.sink_size, N)))
        win = list(range(max(0, N - w), N))
        pinned = set(sink) | set(win)
        rest = sorted([i for i in range(N) if i not in pinned],
                      key=lambda i: imp[i].item(), reverse=True)
        budget_extra = max(0, self.budget - len(pinned))
        idx = sorted(pinned | set(rest[:budget_extra]))

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

        # permanent eviction of lowest cum-attn unprotected token
        while len(self.pos) > self.budget:
            max_pos = max(self.pos)
            cand = self.cum.clone()
            for i, p in enumerate(self.pos):
                if p < self.sink_size or p > max_pos - self.window_size:
                    cand[i] = float("inf")
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
