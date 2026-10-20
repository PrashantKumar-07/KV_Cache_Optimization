"""Opt-in GQA-head residency ablation; GPU research path, full prefill only.

Reuse the cache policy per KV head and repack in the model's original order.
Payload capacity is unchanged; metadata/temporary allocation and Python work
increase. Counters describe head-row events, not whole-layer token events.
Head-specific selection is established prior work, not a novelty claim.
"""
import torch
from longbench_eval import TieredKVPolicy


class GroupedRetentionPolicy(TieredKVPolicy):
    name = 'TieredKV (GQA-head residency prototype)'

    def __init__(self, **options):
        if options.get('prefill_chunk_size', 0):
            raise ValueError('Grouped residency requires full prefill')
        super().__init__(**options)
        self._head_counts = []

    def reset(self):
        super().reset()
        self._head_counts = []

    def _flatten(self, past):
        if len(past) != len(self._head_counts):
            raise ValueError('Model layer count changed')
        flat = []
        for (keys, values), heads in zip(past, self._head_counts):
            if keys.ndim != 4 or keys.shape != values.shape or keys.shape[0] != 1 or keys.shape[1] != heads:
                raise ValueError('Require batch-one unchanged KV head geometry')
            flat.extend((keys[:, head:head+1], values[:, head:head+1]) for head in range(heads))
        return tuple(flat)

    def _pack(self, flat):
        if len(flat) != sum(self._head_counts):
            raise ValueError('Virtual head cache count changed')
        result = []; start = 0
        for heads in self._head_counts:
            group = flat[start:start+heads]
            if len({pair[0].shape[2] for pair in group}) != 1:
                raise ValueError('Head visibility widths differ')
            result.append(tuple(torch.cat([pair[channel] for pair in group], dim=1)
                                for channel in (0, 1)))
            start += heads
        return tuple(result)

    def _bifurcate(self, past, **positions):
        if any(value is not None for value in positions.values()):
            raise ValueError('Grouped residency does not support bounded-prefill metadata')
        if not past or set(self.prompt_q_by_layer) != set(range(len(past))):
            raise ValueError('Actual prompt queries required for every model layer')
        self._head_counts = [keys.shape[1] for keys, _ in past]
        captured = self.prompt_q_by_layer
        grouped = {}; start = 0
        for layer, heads in enumerate(self._head_counts):
            queries = captured[layer]
            if heads < 1 or queries.ndim != 3 or queries.shape[0] < 1 or queries.shape[0] % heads:
                raise ValueError('Invalid captured GQA query geometry')
            per_head = queries.shape[0] // heads
            for head in range(heads):
                grouped[start+head] = queries[head*per_head:(head+1)*per_head]
            start += heads
        self.prompt_q_by_layer = grouped
        visible = super()._bifurcate(self._flatten(past))
        start = 0
        for heads in self._head_counts:
            for head in range(heads):
                # Global prefill sums query heads, decode averages them.
                # Keep the same prior/observation scale in each virtual head.
                self.layer_caches[start+head].sram_cum.mul_(heads)
            start += heads
        return self._pack(visible)

    def _decode_step(self, past):
        return self._pack(super()._decode_step(self._flatten(past)))

    def update_scores(self, past, weights):
        if weights is None:
            return
        if len(weights) != len(self._head_counts):
            raise ValueError('Require attention feedback for every model layer')
        split = []
        for weight, heads in zip(weights, self._head_counts):
            if weight is None or weight.ndim != 4 or weight.shape[0] != 1 or weight.shape[2] != 1 or weight.shape[1] < 1 or weight.shape[1] % heads:
                raise ValueError('Grouped residency supports single-token GQA feedback')
            per_head = weight.shape[1] // heads
            split.extend(weight[:, head*per_head:(head+1)*per_head] for head in range(heads))
        super().update_scores(self._flatten(past), split)
