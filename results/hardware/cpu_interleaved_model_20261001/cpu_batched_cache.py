"""Opt-in cross-layer pinned cold transfers, preserving per-layer policy."""
import torch

from cpu_slot_cache import CpuSlotCache, CpuSlotPolicy, transfer_kernels
from longbench_eval import TieredKVPolicy


class SharedPool:
    def __init__(self, caches):
        self.caches = dict(caches)
        configs = {(c.cfg.num_heads, c.cfg.head_dim, c.capacity,
                    c.cfg.resolved_stt_kv_dtype, str(c.cfg.device)) for c in caches.values()}
        if len(configs) != 1:
            raise ValueError('Shared pool needs equal layer geometry, precision and device')
        self.d2h, self.h2d = transfer_kernels()
        self.pending = []
        self.pending_slots = set()
        self.last_use = None
        self.d2h_calls = self.h2d_calls = 0
        first = next(iter(caches.values()))
        self.device, self.dim = first.cfg.device, first.cfg.head_dim
        span = first.cfg.num_heads * first.capacity
        self.pool = torch.empty((2, len(caches) * span, 1, self.dim),
                                dtype=first.cfg.resolved_stt_kv_dtype, pin_memory=True)
        for number, cache in enumerate(caches.values()):
            if cache._last_use is not None:
                cache._last_use.synchronize()
            block = self.pool[:, number * span:(number + 1) * span]
            block.copy_(cache.pool)
            cache.pool = block
            cache.base = number * span
            cache.manager = self
            cache._last_use = None

    def wait(self):
        if self.last_use is not None:
            torch.cuda.current_stream(self.device).wait_event(self.last_use)

    def record(self):
        self.last_use = torch.cuda.Event()
        self.last_use.record(torch.cuda.current_stream(self.device))
        for cache in self.caches.values():
            cache._last_use = self.last_use

    def enqueue(self, cache, keys, values, slots):
        if not slots:
            return
        addresses = {cache.base + h * cache.capacity + slot
                     for h in range(cache.cfg.num_heads) for slot in slots}
        # A released slot can be reused before a diagnostic/boundary flush.
        # Preserve submitted write order rather than racing duplicate destinations.
        if addresses & self.pending_slots:
            self.flush()
        packed = torch.stack((keys, values)).to(cache.cfg.resolved_stt_kv_dtype).contiguous()
        self.pending.append((cache, packed, list(slots)))
        self.pending_slots.update(addresses)
        cache.host_write_bytes += packed.numel() * packed.element_size()

    def flush(self):
        if not self.pending:
            return
        with torch.cuda.device(self.device):
            self.wait()
            packed = torch.cat([data.reshape(2, -1, 1, self.dim)
                                for _, data, _ in self.pending], dim=1)
            destination = torch.tensor([cache.base + h * cache.capacity + slot
                                        for cache, _, slots in self.pending
                                        for h in range(cache.cfg.num_heads) for slot in slots],
                                       dtype=torch.int32, device=self.device)
            source = torch.arange(packed.shape[1], dtype=torch.int32, device=self.device)
            self.d2h.kv_cache_d2h_transfer(packed, self.pool, source, destination)
            self.record()
            self.d2h_calls += 1
            for cache, _, _ in self.pending:
                cache.d2h_calls += 1  # contributing writes; physical dispatch count is above
            self.pending.clear()
            self.pending_slots.clear()

    def read(self, selections):
        self.flush()
        lengths = {layer: self.caches[layer].cfg.num_heads * len(idx)
                   for layer, idx in selections.items()}
        with torch.cuda.device(self.device):
            self.wait()
            source = torch.tensor([cache.base + h * cache.capacity + cache._logical[i]
                                   for layer, indices in selections.items()
                                   for cache in (self.caches[layer],)
                                   for h in range(cache.cfg.num_heads) for i in indices],
                                  dtype=torch.int32, device=self.device)
            count = sum(lengths.values())
            rows = torch.empty((2, count, 1, self.dim), device=self.device, dtype=self.pool.dtype)
            if count:
                # Native module owns a single mapped pointer; register this pool per dispatch.
                self.h2d.register_cpu_kv_cache_pinned(self.pool)
                destination = torch.arange(count, dtype=torch.int32, device=self.device)
                self.h2d.kv_cache_h2d_transfer(self.pool, rows, source, destination)
                self.h2d_calls += 1
                self.record()
            outputs, offset = {}, 0
            for layer, indices in selections.items():
                cache, length = self.caches[layer], lengths[layer]
                pair = rows[:, offset:offset + length].reshape(2, cache.cfg.num_heads, len(indices), self.dim).to(cache.cfg.resolved_kv_dtype)
                outputs[layer] = (pair[0], pair[1])
                cache.host_read_bytes += length * 2 * self.dim * self.pool.element_size()
                if length:
                    cache.h2d_calls += 1  # contributing reads, distinct from physical dispatches
                offset += length
            return outputs

    def synchronize(self):
        self.flush()
        if self.last_use is not None:
            self.last_use.synchronize()

    def close(self):
        self.synchronize()
        for cache in self.caches.values():
            cache.manager = None
            cache._prefetched = None
        self.caches.clear()
        self.pool = None
        self.last_use = None


class CpuBatchCache(CpuSlotCache):
    def __init__(self, cfg):
        self.manager = None
        self._prefetched = None
        self.base = 0
        super().__init__(cfg)

    def _write(self, keys, values, slots):
        if self.manager is None:
            return super()._write(keys, values, slots)
        self.manager.enqueue(self, keys, values, slots)

    def _canonical(self, channel):
        if self.manager is not None:
            self.manager.synchronize()
        return super()._canonical(channel)

    def select_stt_pair(self, idx):
        indices = list(idx)
        if self._prefetched is not None:
            planned, pair = self._prefetched
            if indices != planned:
                raise RuntimeError('Cold row identities changed between planning and consumption')
            self._prefetched = None
            return pair
        if self.manager is None:
            return super().select_stt_pair(indices)
        layer = next(layer for layer, cache in self.manager.caches.items() if cache is self)
        return self.manager.read({layer: indices})[layer]

    def close(self):
        if self.manager is not None:
            self.manager.synchronize()
        self._prefetched = None
        super().close()


class CpuBatchPolicy(CpuSlotPolicy):
    def __init__(self, **kwargs):
        self.manager = None
        self._planned_promotions = None
        super().__init__(**kwargs)
        self.cache_cls = CpuBatchCache

    def _ensure_manager(self):
        if self.manager is None and self.layer_caches:
            self.manager = SharedPool(self.layer_caches)

    def _prefetch(self, selections):
        self._ensure_manager()
        if self.manager is None:
            return
        if any(cache._prefetched is not None for cache in self.layer_caches.values()):
            raise RuntimeError('Unconsumed prefetched cold rows')
        pairs = self.manager.read(selections)
        for layer, indices in selections.items():
            self.layer_caches[layer]._prefetched = (list(indices), pairs[layer])

    def _prepare_visible_reads(self, selections):
        self._prefetch(selections)

    def _prepare_decode_reads(self):
        self._planned_promotions = {layer: TieredKVPolicy._promotion_plan(self, layer, cache)
                                    for layer, cache in self.layer_caches.items()}
        selections = {}
        for layer, (hits, _) in self._planned_promotions.items():
            if hits:
                indices, _ = self.layer_caches[layer].promotion_rows(hits)
                if indices:
                    selections[layer] = indices
        self._prefetch(selections)

    def _promotion_plan(self, layer_idx, cache):
        if self._planned_promotions is None:
            return super()._promotion_plan(layer_idx, cache)
        return self._planned_promotions.pop(layer_idx)

    def reset(self):
        if self.manager is not None:
            self.manager.close()
            self.manager = None
        self._planned_promotions = None
        super().reset()
