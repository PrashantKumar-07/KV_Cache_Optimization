# attention.py — core attention math shared by all cache variants

from __future__ import annotations
import math
import torch


def scaled_dot_product_attention(q, k, v, causal=False):
    """Standard scaled dot-product attention with MAC counting.

    Args:
        q: (H, Lq, D)
        k: (H, Lk, D)
        v: (H, Lk, D)
        causal: apply upper-triangular mask (prefill only)

    Returns:
        out          (H, Lq, D)
        attn_weights (H, Lq, Lk)
        macs         int — multiply-accumulate ops
    """
    H, Lq, D = q.shape
    Lk = k.shape[1]
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (H, Lq, Lk)

    if causal and Lq == Lk:
        mask = torch.triu(torch.ones(Lq, Lk, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)

    macs = 2 * H * Lq * Lk * D  # QK^T + weights*V
    return out, attn_weights, macs


def snapkv_importance(q_window, k_all, pool_kernel=5):
    """SnapKV importance scores for all N past tokens.

    Computes attention from the last `window` queries against all keys,
    sums received attention mass per key, then average-pools over the token
    axis so selection favours contiguous clusters rather than isolated spikes.

    Args:
        q_window: (H, W, D) — last W query vectors (observation window)
        k_all:    (H, N, D) — full key cache
        pool_kernel: pooling kernel size

    Returns:
        importance: (N,) float — higher = more important
    """
    H, W, D = q_window.shape
    N = k_all.shape[1]
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q_window, k_all.transpose(-2, -1)) * scale  # (H, W, N)
    attn = torch.softmax(scores, dim=-1)
    importance = attn.sum(dim=1).sum(dim=0)  # sum over window and heads -> (N,)

    if pool_kernel > 1 and N >= pool_kernel:
        pad = pool_kernel // 2
        pooled = torch.nn.functional.avg_pool1d(
            importance.view(1, 1, N), kernel_size=pool_kernel, stride=1, padding=pad
        ).view(-1)
        importance = pooled[:N]

    return importance
