"""Incremental page sums must match full rebuilds after cache transitions."""
import sys
from pathlib import Path

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src', 'experiments')]
from tiered_kv_cache import TieredConfig
from stable_page_index import IndexedTieredKVCache


@pytest.mark.parametrize('inclusive,slow_capacity', [(False, 48), (True, 48), (True, 96)])
def test_incremental_centroids_follow_live_kv(inclusive, slow_capacity):
    torch.manual_seed(1)
    cfg = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                       sram_capacity=32, sttram_capacity=slow_capacity, page_size=4,
                       promote_top_pages=2, inclusive=inclusive)
    cache = IndexedTieredKVCache(cfg)
    k, v = torch.randn(2, 96, 8), torch.randn(2, 96, 8)
    cache.initial_bifurcation(k, v, k)
    cache.page_index.validate(cache)
    for step in range(100):
        q = (3 * k[:, 2 + step % 38:3 + step % 38]
             if step % 2 == 0 else torch.randn(2, 1, 8))
        cache.step(q, torch.randn(2, 1, 8), torch.randn(2, 1, 8), 96 + step)
        cache.page_index.validate(cache)
        live = torch.where(~cache.stt_shadow)[0]
        if len(live) >= 8:
            rows = cache.page_index.candidate_rows(q[:, 0].unsqueeze(1), cache,
                                                   live, page_count=2, min_rows=8)
            assert len(rows) >= 8
            assert rows.unique().numel() == len(rows)
            assert (rows < len(live)).all()
    assert cache.metrics.total_promoted > 0
    assert cache.page_index.changed_key_bytes > 0
    assert cache.metrics.total_dropped > 0
    if inclusive and slow_capacity == 96:
        assert cache.metrics.total_writes_saved > 0
        assert sum(step.backups_reclaimed for step in cache.metrics.steps) > 0
