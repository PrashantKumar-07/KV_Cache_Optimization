"""Opt-in pinned CPU-DRAM cold storage; unchanged TieredKV policy semantics."""
from functools import lru_cache
import json
from pathlib import Path

import torch

from longbench_eval import TieredKVPolicy
from probe_cpu_slots import load_kernel
from tiered_kv_cache import TieredKVCache


@lru_cache(maxsize=1)
def transfer_kernels():
    directory = Path('/data/nishant/tieredkv_external/flexicache_transfer_build_20261001')
    d2h, dhash = load_kernel(directory, 'd2h')
    h2d, hhash = load_kernel(directory, 'h2d')
    expected = json.loads((Path(__file__).resolve().parents[1] / 'results/baseline_fidelity/flexicache_transfer_check_v2_20261001.json').read_text())['binary_sha256']
    if {'d2h': dhash, 'h2d': hhash} != expected:
        raise ValueError('Native transfer binaries differ from the verified source build')
    return d2h, h2d


class CpuSlotCache(TieredKVCache):
    """Physical host slots, with canonical logical tensor snapshots for inspection.

    Internal paired hooks avoid snapshots. Explicit stt_k/stt_v inspection
    synchronizes pending host writes and gathers logical rows on CPU.
    """

    def __init__(self, cfg):
        if torch.device(cfg.device).type != 'cuda':
            raise ValueError('Native host slots require CUDA')
        if cfg.resolved_stt_kv_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('Native host slots support fp16/bf16 storage only')
        self.pool = None
        self._last_use = None
        self._logical = []
        self._free = []
        super().__init__(cfg)
        self.capacity = cfg.sttram_capacity + cfg.sram_capacity + 1
        self.host_read_bytes = 0
        self.host_write_bytes = 0
        self.host_initial_write_bytes = 0
        self.d2h_calls = 0
        self.h2d_calls = 0
        self._d2h, self._h2d = transfer_kernels()

    @property
    def stt_k(self):
        return self._canonical(0) if self.pool is not None else self._raw_k

    @stt_k.setter
    def stt_k(self, value):
        if self.pool is not None:
            raise RuntimeError('Use paired storage hooks for an active slot pool')
        self._raw_k = value

    @property
    def stt_v(self):
        return self._canonical(1) if self.pool is not None else self._raw_v

    @stt_v.setter
    def stt_v(self, value):
        if self.pool is not None:
            raise RuntimeError('Use paired storage hooks for an active slot pool')
        self._raw_v = value

    def _canonical(self, channel):
        if self._last_use is not None:
            self._last_use.synchronize()
        return self.pool[channel].reshape(self.cfg.num_heads, self.capacity, self.cfg.head_dim).index_select(1, torch.tensor(self._logical, dtype=torch.long))

    def _wait(self):
        if self._last_use is not None:
            torch.cuda.current_stream(self.cfg.device).wait_event(self._last_use)

    def _record(self):
        self._last_use = torch.cuda.Event()
        self._last_use.record(torch.cuda.current_stream(self.cfg.device))

    def _blocks(self, slots):
        return torch.tensor([head * self.capacity + slot
                             for head in range(self.cfg.num_heads) for slot in slots],
                            dtype=torch.int32, device=self.cfg.device)

    def _write(self, keys, values, slots):
        if not slots:
            return
        with torch.cuda.device(self.cfg.device):
            self._wait()
            packed = torch.stack((keys, values)).to(self.cfg.resolved_stt_kv_dtype).contiguous()
            source = torch.arange(self.cfg.num_heads * len(slots), device=self.cfg.device, dtype=torch.int32)
            self._d2h.kv_cache_d2h_transfer(packed.reshape(2, -1, 1, self.cfg.head_dim), self.pool, source, self._blocks(slots))
            self._record()
        self.host_write_bytes += packed.numel() * packed.element_size()
        self.d2h_calls += 1

    def initial_bifurcation(self, *args, **kwargs):
        self.close()
        counts = super().initial_bifurcation(*args, **kwargs)
        keys, values = self._raw_k, self._raw_v
        count = len(self.stt_pos)
        if count > self.capacity:
            raise ValueError('Initial cold rows exceed reserved slot capacity')
        self.pool = torch.empty((2, self.cfg.num_heads * self.capacity, 1, self.cfg.head_dim), dtype=self.cfg.resolved_stt_kv_dtype, pin_memory=True)
        self._logical = list(range(count))
        self._free = list(range(count, self.capacity))
        self._write(keys, values, self._logical)
        self.host_initial_write_bytes = self.host_write_bytes
        self._raw_k = self._raw_v = None
        return counts

    def select_stt_pair(self, idx):
        if self.pool is None:
            return super().select_stt_pair(idx)
        slots = [self._logical[i] for i in idx]
        with torch.cuda.device(self.cfg.device):
            self._wait()
            rows = torch.empty((2, self.cfg.num_heads * len(slots), 1, self.cfg.head_dim), dtype=self.cfg.resolved_stt_kv_dtype, device=self.cfg.device)
            if slots:
                self._h2d.register_cpu_kv_cache_pinned(self.pool)
                dest = torch.arange(self.cfg.num_heads * len(slots), device=self.cfg.device, dtype=torch.int32)
                self._h2d.kv_cache_h2d_transfer(self.pool, rows, self._blocks(slots), dest)
                self.h2d_calls += 1
                self._record()
        self.host_read_bytes += rows.numel() * rows.element_size()
        rows = rows.reshape(2, self.cfg.num_heads, len(slots), self.cfg.head_dim).to(self.cfg.resolved_kv_dtype)
        return rows[0], rows[1]

    def select_stt(self, tensor, idx):
        rows = self.index_stt_storage(tensor, idx)
        self.host_read_bytes += rows.numel() * rows.element_size()
        return rows.to(device=self.cfg.device, dtype=self.cfg.resolved_kv_dtype)

    def append_stt_pair(self, keys, values):
        count = keys.shape[1]
        if count > len(self._free):
            raise ValueError('Cold append exceeds reserved transient slot capacity')
        slots = self._free[:count]
        self._write(keys, values, slots)
        self._free = self._free[count:]
        self._logical.extend(slots)

    def compact_stt_pair(self, keep):
        live = set(keep)
        self._free.extend(slot for i, slot in enumerate(self._logical) if i not in live)
        self._logical = [self._logical[i] for i in keep]

    def sketch_key_tensor(self):
        return self.select_stt_pair(range(len(self.stt_pos)))[0]

    @property
    def reserved_host_bytes(self):
        return 0 if self.pool is None else self.pool.numel() * self.pool.element_size()

    @property
    def occupied_host_bytes(self):
        return len(self._logical) * 2 * self.cfg.num_heads * self.cfg.head_dim * torch.empty((), dtype=self.cfg.resolved_stt_kv_dtype).element_size()

    def close(self):
        if self._last_use is not None:
            self._last_use.synchronize()
        self._last_use = None
        self.pool = None
        self._logical = []
        self._free = []

    def __del__(self):
        # Explicit policy reset owns normal cleanup; guard exceptional exits too.
        try:
            self.close()
        except Exception:
            pass


class CpuSlotPolicy(TieredKVPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_cls = CpuSlotCache

    def reset(self):
        outgoing = list(self.layer_caches.values())
        super().reset()
        for cache in outgoing:
            cache.close()
