"""Offline CPU smoke check: actual prefill, decode, tier bounds and accounting."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from tiered_kv_cache import TieredConfig, TieredKVCache


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = TieredConfig(num_heads=2, head_dim=8, sink_size=2, window_size=4,
                         sram_capacity=16, sttram_capacity=32, page_size=4,
                         promote_top_pages=1, sttram_bifurcation_frac=0.75)
    cache = TieredKVCache(config)
    k, v, q = (torch.randn(2, 80, 8) for _ in range(3))
    cache.initial_bifurcation(k, v, q)
    for position in range(80, 96):
        query, key, value = (torch.randn(2, 1, 8) for _ in range(3))
        output = cache.step(query, key, value, position)
        assert output.shape == (2, 1, 8) and torch.isfinite(output).all()
        assert len(cache.sram_pos) <= config.sram_capacity
        assert len(cache.stt_pos) <= config.sttram_capacity
    metrics = cache.metrics
    assert len(metrics.steps) == 16
    assert metrics.total_paid_writes + metrics.total_writes_saved == metrics.total_demoted
    print(json.dumps({"passed": True, "device": "cpu", "decode_steps": 16,
                      "metrics": metrics.summary(),
                      "scope": "random tensors; simulated traffic costs; no model quality claim"},
                     indent=2))


if __name__ == "__main__":
    main()
