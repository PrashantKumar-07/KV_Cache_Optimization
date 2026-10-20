"""Opt-in bounded transient slots; retain the original logical cache policy."""
from functools import partial
from direct_shared_pool import DirectCpuCache, DirectScorePolicy


class BoundedCpuCache(DirectCpuCache):
    def __init__(self, cfg, extra_slots):
        super().__init__(cfg)
        self.capacity = cfg.sttram_capacity + extra_slots


class BoundedPoolPolicy(DirectScorePolicy):
    def __init__(self, *, max_decode_append=32, **options):
        super().__init__(**options)
        budget = self.promotion_budget
        if type(max_decode_append) is not int or max_decode_append < 1:
            raise ValueError('max_decode_append must be a positive integer')
        if type(budget) is not int or budget + max_decode_append > self.sram_budget + 1:
            raise ValueError('Use the original pool for this promotion/append geometry')
        if self.sram_budget < self.start_size + self.recent_size:
            raise ValueError('Hard protection exceeds the supported steady hot capacity')
        self.max_decode_append = max_decode_append
        self.reserved_promotion_budget = budget
        self.direct_cache_cls = partial(BoundedCpuCache, extra_slots=budget+max_decode_append)

    def _decode_step(self, past):
        # Validate before decode counters, hysteresis, promotion or eviction mutate.
        layers = list(self.layer_caches)
        if not layers or layers != list(range(len(past))):
            raise ValueError('Bounded pools require complete ordered layer geometry')
        if type(self.promotion_budget) is not int or not 0 <= self.promotion_budget <= self.reserved_promotion_budget:
            raise ValueError('Promotion budget exceeds the reserved transient bound')
        appended = [max(1,past[i][0].size(2)-self._vis_len[i]) for i in layers]
        if (len(set(appended)) != 1 or appended[0] > self.max_decode_append or
            any(c.cfg.sttram_capacity+self.promotion_budget+appended[0] > c.capacity
                or c.capacity != self.manager.geometry[1] for c in self.layer_caches.values())):
            raise ValueError('Decode append exceeds the reserved transient bound')
        if any(len(c.sram_pos)>c.cfg.sram_capacity or len(c.stt_pos)>c.cfg.sttram_capacity
               for c in self.layer_caches.values()):
            raise ValueError('Steady cache occupancy exceeds the transient-bound assumptions')
        return super()._decode_step(past)
