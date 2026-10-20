"""Synthetic full-cache and native alternating-pool parity before model trials."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from cpu_slot_cache import CpuSlotCache, CpuSlotPolicy
from probe_cpu_cold import CpuColdIndexSelectCache
from provenance import atomic_json, runtime_identity
from tiered_kv_cache import TieredConfig


FIELDS = ('sram_k', 'sram_v', 'sram_pos', 'sram_cum', 'sram_age', 'sram_grace',
          'stt_k', 'stt_v', 'stt_pos', 'stt_last_access', 'stt_shadow',
          'stt_was_vram_resident', 'stt_peek_streak')


def check_state(left, right):
    for field in FIELDS:
        a, b = getattr(left, field), getattr(right, field)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a.cpu(), b.cpu()), field
        else:
            assert a == b, field
    assert left.host_read_bytes == right.host_read_bytes, 'read payload'
    assert left.host_write_bytes == right.host_write_bytes, 'write payload'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = {'runtime': runtime_identity(), 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'backend_sha256': hashlib.sha256(Path('experiments/cpu_slot_cache.py').read_bytes()).hexdigest(), 'passed': False, 'checks': []}
    owned = []
    try:
        torch.manual_seed(0)
        for inclusive in (True, False):
            for fill in (.75, 1.):
                cfg = TieredConfig(num_heads=8, head_dim=128, sink_size=2, window_size=4,
                    sram_capacity=32, sttram_capacity=64, page_size=4,
                    kv_dtype=torch.bfloat16, device='cuda', inclusive=inclusive,
                    sttram_bifurcation_frac=fill)
                pairs = []
                for layer in range(2):
                    reference, candidate = CpuColdIndexSelectCache(cfg), CpuSlotCache(cfg)
                    owned.append(candidate)
                    k, v, q = (torch.randn(8, 256, 128, device='cuda') for _ in range(3))
                    assert reference.initial_bifurcation(k, v, q) == candidate.initial_bifurcation(k, v, q)
                    check_state(reference, candidate)
                    pairs.append((reference, candidate))
                for step in range(24):
                    for reference, candidate in pairs:
                        m = len(reference.stt_pos)
                        idx = [i for i in range(m) if not reference.stt_shadow[i]][:8]
                        ah, ap = reference.resolve_promotion_indices(idx)
                        bh, bp = candidate.resolve_promotion_indices(idx)
                        assert (ah, ap) == (bh, bp)
                        # Read before promotion, including empty lists and repeated selections.
                        for selected in ([], idx, idx[::-1], idx[:1] * 2):
                            a, b = reference.select_stt_pair(selected), candidate.select_stt_pair(selected)
                            assert all(torch.equal(x, y) for x, y in zip(a, b))
                        a, b = reference.get_peek_kv([(i, i + 1) for i in ap]), candidate.get_peek_kv([(i, i + 1) for i in bp])
                        assert a[2] == b[2]
                        if a[0] is not None:
                            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                        assert reference.promote([(i, i + 1) for i in ah]) == candidate.promote([(i, i + 1) for i in bh])
                        count = len(reference.sram_pos)
                        scores = torch.arange(count, device='cuda', dtype=torch.float32) % 7
                        for cache in (reference, candidate):
                            cache.sram_cum.copy_(scores)
                            cache.sram_age.fill_(step + 1)
                            cache._t = step + 1
                        k, v = (torch.randn(8, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2))
                        assert reference.evict_and_demote(k, v, 256 + step) == candidate.evict_and_demote(k, v, 256 + step)
                        check_state(reference, candidate)
                # Verify canonical sketch keys without charging this diagnostic to the control.
                for reference, candidate in pairs:
                    assert torch.equal(candidate.sketch_key_tensor().cpu(), reference.stt_k)
                    assert len(candidate._logical) == len(candidate.stt_pos)
                    assert len(set(candidate._logical + candidate._free)) == candidate.capacity
                    assert candidate.occupied_host_bytes <= candidate.reserved_host_bytes
                    before = list(candidate._logical)
                    try:
                        huge = torch.empty(8, candidate.capacity + 1, 128, device='cuda', dtype=torch.bfloat16)
                        candidate.append_stt_pair(huge, huge)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('capacity guard missing')
                    assert candidate._logical == before
                    candidate.close()
                    candidate.close()
                    assert candidate.pool is None and candidate._last_use is None
                report['checks'].append({'inclusive': inclusive, 'initial_fill': fill, 'layers': 2, 'steps': 24, 'exact_state_payload_and_policy': True, 'close_and_capacity_guard': True})
        # Reset must close pools even with an asynchronous host write pending.
        policy = CpuSlotPolicy(sram_budget=32, stt_budget=64)
        cache = CpuSlotCache(cfg)
        reset_k, reset_v = (torch.randn(8, 256, 128, device="cuda") for _ in range(2))
        cache.initial_bifurcation(reset_k, reset_v, reset_k)
        assert cache._last_use is not None and cache.host_initial_write_bytes > 0
        policy.layer_caches[0] = cache
        policy.reset()
        assert cache.pool is None and not policy.layer_caches
        report.update(passed=True, reset_completion=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for cache in owned:
            cache.close()
        atomic_json(args.output, report)
    print(json.dumps({'passed': report['passed'], 'checks': len(report['checks'])}))


if __name__ == '__main__':
    main()
