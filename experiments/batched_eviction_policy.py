"""Opt-in cross-layer promotion preparation and exact batched hot eviction."""
from types import SimpleNamespace
import torch
from direct_shared_pool import DirectCpuCache, DirectScorePolicy
from probe_batched_eviction_selection import batched_victims


class BatchedVictimCache(DirectCpuCache):
    def __init__(self,cfg):
        self.prepared_promotion = self.prepared_victims = None
        super().__init__(cfg)

    def promote(self,hits):
        if self.prepared_promotion is None:
            return super().promote(hits)
        expected,count = self.prepared_promotion
        self.prepared_promotion = None
        if hits != expected:
            raise RuntimeError('Promotion identities changed after cross-layer preparation')
        return count

    def _select_hot_victims(self):
        if self.prepared_victims is None:
            return super()._select_hot_victims()
        positions,victims = self.prepared_victims
        self.prepared_victims = None
        if self.sram_pos != positions:
            raise RuntimeError('Hot positions changed after cross-layer eviction planning')
        return victims

    def close(self):
        self.prepared_promotion = self.prepared_victims = None
        super().close()


class BatchedEvictionPolicy(DirectScorePolicy):
    direct_cache_cls = BatchedVictimCache

    def _decode_step(self,past):
        self.incoming_past = past
        try:
            result = super()._decode_step(past)
            if any(c.prepared_promotion is not None or c.prepared_victims is not None
                   for c in self.layer_caches.values()):
                raise RuntimeError('Unconsumed cross-layer eviction plan')
            return result
        finally:
            self.incoming_past = None

    def _prepare_decode_reads(self):
        super()._prepare_decode_reads()
        past = self.incoming_past
        layers = list(self.layer_caches)
        if layers != list(range(len(past))) or any(c.cfg.write_aware_lambda != 0 or
            c.sram_cum.dtype != torch.float32 or c.sram_age.dtype != torch.float32 or
            c.sram_cum.device.type != 'cuda' for c in self.layer_caches.values()):
            return  # Preserve the original per-layer path for unsupported configurations.
        first = layers[0] if layers else None
        n_new = past[0][0].size(2)-self._vis_len[first] if first in self._vis_len else 1
        n_new = max(1,n_new)
        projections = []
        for layer,cache in self.layer_caches.items():
            if cache.prepared_promotion is not None or cache.prepared_victims is not None:
                raise RuntimeError('Stale cross-layer eviction plan')
            hits,_ = self._planned_promotions[layer]
            if hits is not None:
                # The original layer loop increments _t immediately before promotion.
                cache._t += 1
                try:
                    promoted = cache.promote(hits)
                finally:
                    cache._t -= 1
                cache.prepared_promotion = (list(hits),promoted)
            start = cache.metrics.prompt_len+self.decode_step_count-n_new
            positions = cache.sram_pos+list(range(start,start+n_new))
            zeros = torch.zeros(n_new,device=cache.cfg.device,dtype=cache.cfg.dtype)
            projections.append(SimpleNamespace(cfg=cache.cfg,sram_pos=positions,
                sram_cum=torch.cat((cache.sram_cum,zeros)),
                sram_age=torch.cat((cache.sram_age+1,zeros)),
                sram_grace=torch.cat((cache.sram_grace,torch.zeros(n_new,dtype=torch.bool)))))
        selected = batched_victims(projections,numpy_masks=True)
        for cache,projection,victims in zip(self.layer_caches.values(),projections,selected):
            cache.prepared_victims = (projection.sram_pos,victims)

    def reset(self):
        self.incoming_past = None
        super().reset()
