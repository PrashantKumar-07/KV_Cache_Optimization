"""
attention.py
------------
Shared scaled-dot-product attention utilities, written to mirror the math in
"Attention Is All You Need" exactly, plus explicit MAC (multiply-accumulate)
counting so every experiment reports real compute cost, not an estimate.

All tensors use the convention:  (num_heads, seq_len, head_dim)   ==  (H, N, D)
Queries for a single decode step:  (H, 1, D)

Nothing here is model-specific; the same routine serves the full-attention
oracle, the baselines, and the tiered cache, so any accuracy gap comes purely
from *which* K/V tokens are visible, never from a different attention formula.
"""

from __future__ import annotations
import math
import torch


def scaled_dot_product_attention(
    q: torch.Tensor,          # (H, Lq, D)
    k: torch.Tensor,          # (H, Lk, D)
    v: torch.Tensor,          # (H, Lk, D)
    causal: bool = False,
):
    """
    Returns
    -------
    out          : (H, Lq, D)   attention output
    attn_weights : (H, Lq, Lk)  softmax weights (row i sums to 1)
    macs         : int          multiply-accumulate ops for this call

    Implements  softmax(Q Kᵀ / sqrt(d_k)) V  with an optional causal mask.
    """
    H, Lq, D = q.shape
    Lk = k.shape[1]
    scale = 1.0 / math.sqrt(D)

    # scores = Q Kᵀ / sqrt(d_k)      -> (H, Lq, Lk)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal and Lq == Lk:
        # upper triangle (future positions) set to -inf before softmax
        mask = torch.triu(torch.ones(Lq, Lk, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)           # (H, Lq, Lk)
    out = torch.matmul(attn_weights, v)                    # (H, Lq, D)

    # MACs: QKᵀ = H*Lq*Lk*D  ;  weights·V = H*Lq*Lk*D
    macs = 2 * H * Lq * Lk * D
    return out, attn_weights, macs


def snapkv_importance(
    q_window: torch.Tensor,   # (H, W, D)  the last W queries (observation window)
    k_all: torch.Tensor,      # (H, N, D)  keys for every past token
    pool_kernel: int = 5,
):
    """
    SnapKV-style importance score for every one of the N past tokens.

    1. Attention of the W observation-window queries against all N keys.
    2. Sum the attention mass each key receives across the window and heads.
    3. 1-D average pool (kernel=pool_kernel) so a selected token drags in its
       neighbours -- this is what makes SnapKV robust (clustered, not spiky).

    Returns importance : (N,) float tensor, higher = more important.
    """
    H, W, D = q_window.shape
    N = k_all.shape[1]
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q_window, k_all.transpose(-2, -1)) * scale  # (H, W, N)
    attn = torch.softmax(scores, dim=-1)                             # (H, W, N)

    # attention mass received by each key, summed over window and heads
    importance = attn.sum(dim=1).sum(dim=0)                          # (N,)

    # average pool over the token axis (clustered selection)
    if pool_kernel > 1 and N >= pool_kernel:
        pad = pool_kernel // 2
        pooled = torch.nn.functional.avg_pool1d(
            importance.view(1, 1, N), kernel_size=pool_kernel, stride=1, padding=pad
        ).view(-1)
        # avg_pool1d with symmetric padding can return N or N+1; trim to N
        importance = pooled[:N]

    return importance
