"""Opt-in homogeneous cross-layer score bookkeeping, exact-policy fallback."""
import torch

from cpu_batched_cache import CpuBatchPolicy
from longbench_eval import TieredKVPolicy


def batched_update(policy, past, weights):
    if weights is None:
        return
    layers = list(range(len(past)))
    # Restrict the prototype to complete ordered hot visibility and equal geometry.
    if not layers or len(weights) != len(layers):
        return TieredKVPolicy.update_scores(policy, past, weights)
    shape = None
    hot = cold = None
    for layer in layers:
        cache = policy.layer_caches.get(layer)
        keep = (policy._vis_sram_keep or {}).get(layer)
        positions = (policy._vis_stt_pos or {}).get(layer)
        weight = weights[layer]
        if cache is None or keep is None or positions is None or weight is None:
            return TieredKVPolicy.update_scores(policy, past, weights)
        if keep != list(range(len(cache.sram_cum))) or not keep:
            return TieredKVPolicy.update_scores(policy, past, weights)
        if shape is None:
            shape, hot, cold = weight.shape, len(keep), len(positions)
        if weight.shape != shape or len(keep) != hot or len(positions) != cold or weight.shape[-1] < hot + cold:
            return TieredKVPolicy.update_scores(policy, past, weights)
    packed = torch.stack([weights[layer][0].to(torch.float32).squeeze(1) for layer in layers])
    folded = packed.mean(dim=1)
    torch._foreach_add_([policy.layer_caches[layer].sram_cum for layer in layers],
                        list(folded[:, :hot].unbind()))
    policy._stt_attn, policy._stt_attn_idx = {}, {}
    if cold:
        values = folded[:, hot:hot + cold].tolist()
        for layer, attention in zip(layers, values):
            cache = policy.layer_caches[layer]
            positions = policy._vis_stt_pos[layer]
            indices = (policy._vis_stt_idx or {}).get(layer)
            policy._stt_attn[layer] = dict(zip(positions, attention))
            match = indices is not None and len(indices) == len(positions) and all(
                i < len(cache.stt_pos) and cache.stt_pos[i] == p for i, p in zip(indices, positions))
            if match:
                policy._stt_attn_idx[layer] = dict(zip(indices, attention))
                touched = [i for i, a in zip(indices, attention) if a > 0]
            else:
                lookup = {p: i for i, p in enumerate(cache.stt_pos)}
                touched = [lookup[p] for p, a in zip(positions, attention) if a > 0 and p in lookup]
            if touched:
                cache.stt_last_access[touched] = cache._t


class ScoreBatchPolicy(CpuBatchPolicy):
    update_scores = batched_update
