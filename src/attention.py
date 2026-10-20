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


def snapkv_importance(q_window, k_all, pool_kernel=5, causal=False):
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
    if causal:
        # Observation queries are the final W prompt positions. A query at
        # position i must not score a key at j > i.
        future = torch.arange(N, device=scores.device)[None, :] > \
                 torch.arange(N - W, N, device=scores.device)[:, None]
        scores = scores.masked_fill(future[None], float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    importance = attn.sum(dim=1).sum(dim=0)  # sum over window and heads -> (N,)

    if pool_kernel > 1 and N >= pool_kernel:
        pad = pool_kernel // 2
        pooled = torch.nn.functional.avg_pool1d(
            importance.view(1, 1, N), kernel_size=pool_kernel, stride=1, padding=pad
        ).view(-1)
        importance = pooled[:N]

    return importance


def gqa_causal_importance(q_window, k_all, pool_kernel=5, query_chunk=16):
    """Same causal importance as expanded-head scoring, without copying GQA keys.

    Chunk the observation queries to bound temporary score memory at
    O(query_heads * query_chunk * prompt_tokens). Selection remains global
    over all query heads and observation positions.
    """
    q_heads, width, dim = q_window.shape
    k_heads, tokens, k_dim = k_all.shape
    if dim != k_dim or q_heads % k_heads or width > tokens or query_chunk < 1:
        raise ValueError('Invalid GQA importance dimensions or chunk size')
    groups = q_heads // k_heads
    grouped_q = q_window.reshape(k_heads, groups, width, dim)
    key_t = k_all[:, None].transpose(-2, -1)
    received = torch.zeros(k_heads, groups, tokens, device=q_window.device,
                           dtype=q_window.dtype)
    keys = torch.arange(tokens, device=q_window.device)
    for start in range(0, width, query_chunk):
        end = min(start + query_chunk, width)
        scores = torch.matmul(grouped_q[:, :, start:end], key_t) / math.sqrt(dim)
        query_positions = torch.arange(tokens - width + start,
                                       tokens - width + end, device=q_window.device)
        scores.masked_fill_(keys[None, None, None, :] >
                            query_positions[None, None, :, None], float('-inf'))
        received += torch.softmax(scores, dim=-1).sum(dim=2)
    importance = received.sum(dim=(0, 1))
    if pool_kernel > 1 and tokens >= pool_kernel:
        pad = pool_kernel // 2
        importance = torch.nn.functional.avg_pool1d(
            importance.view(1, 1, tokens), kernel_size=pool_kernel,
            stride=1, padding=pad).view(-1)[:tokens]
    return importance
