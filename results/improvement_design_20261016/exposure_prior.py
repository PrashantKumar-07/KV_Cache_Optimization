"""Development-only calibrated prompt evidence; initial payload selection unchanged.

The canonical actual-Q importance sums attention over query heads and prompt
observations. Decode records the head mean once per forward. Express the prior
in those same units, with its observation count as age. This changes eviction
statistics only; no cold selector, residency capacity, or attention width change.
"""
from longbench_eval import TieredKVPolicy


class ExposureCalibratedPriorMixin:
    def _bifurcate(self, past_kv, **positions):
        geometry = {}
        # Fail before super mutates any layer. Proxy queries have no attested
        # observation-head geometry and are deliberately unsupported here.
        for index, (keys, _) in enumerate(past_kv):
            query = self.prompt_q_by_layer.get(index)
            width = min(self.recent_size, keys.shape[2])
            if (query is None or query.ndim != 3 or query.shape[0] < 1
                    or query.shape[1] != width or query.shape[2] != keys.shape[3]
                    or keys.shape[1] < 1 or query.shape[0] % keys.shape[1]):
                raise ValueError('Calibrated prior requires actual prompt queries for every layer')
            geometry[index] = (query.shape[0], width)
        visible = super()._bifurcate(past_kv, **positions)
        for index, (heads, observations) in geometry.items():
            cache = self.layer_caches[index]
            cache.sram_cum.div_(heads)
            cache.sram_age.fill_(observations)
        return visible


class ExposureCalibratedPriorPolicy(ExposureCalibratedPriorMixin, TieredKVPolicy):
    name = 'TieredKV (exposure-calibrated actual-Q prior, development only)'
