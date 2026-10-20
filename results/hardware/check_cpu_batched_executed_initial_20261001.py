"""Synthetic shared-pool batching parity, reuse ordering and reset proof."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from check_cpu_slots import check_state
from cpu_batched_cache import CpuBatchCache, CpuBatchPolicy, SharedPool
from probe_cpu_cold import CpuColdIndexSelectCache
from provenance import atomic_json, runtime_identity
from tiered_kv_cache import TieredConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = dict(passed=False, runtime=runtime_identity(), checks=[],
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  backend_sha256=hashlib.sha256(Path('experiments/cpu_batched_cache.py').read_bytes()).hexdigest())
    manager = None
    candidates = {}
    try:
        torch.manual_seed(0)
        for inclusive in (True, False):
            for fill in (.75, 1.):
                cfg = TieredConfig(num_heads=8, head_dim=128, sink_size=2, window_size=4,
                    sram_capacity=32, sttram_capacity=64, page_size=4, kv_dtype=torch.bfloat16,
                    device='cuda', inclusive=inclusive, sttram_bifurcation_frac=fill,
                    stt_live_floor=8 if inclusive else 0)
                references, candidates = {}, {}
                for layer in range(2):
                    reference, candidate = CpuColdIndexSelectCache(cfg), CpuBatchCache(cfg)
                    keys, values, query = (torch.randn(8, 256, 128, device='cuda') for _ in range(3))
                    assert reference.initial_bifurcation(keys, values, query) == candidate.initial_bifurcation(keys, values, query)
                    check_state(reference, candidate)
                    references[layer], candidates[layer] = reference, candidate
                manager = SharedPool(candidates)
                for step in range(24):
                    hits = {}
                    for layer, reference in references.items():
                        candidate = candidates[layer]
                        live = [i for i in range(len(reference.stt_pos)) if not reference.stt_shadow[i]][:8 + layer]
                        ah, ap = reference.resolve_promotion_indices(live)
                        bh, bp = candidate.resolve_promotion_indices(live)
                        assert (ah, ap) == (bh, bp)
                        hits[layer] = [(i, i + 1) for i in ah]
                    for mode in ('empty', 'reverse', 'repeat'):
                        selections = {layer: ([] if mode == 'empty' else list(range(min(9, len(c.stt_pos))))[::-1] if mode == 'reverse' else [0, 0] if c.stt_pos else []) for layer, c in candidates.items()}
                        actual = manager.read(selections)
                        for layer, indices in selections.items():
                            expected = references[layer].select_stt_pair(indices)
                            assert all(torch.equal(a, b) for a, b in zip(actual[layer], expected))
                    selections = {layer: c.promotion_rows(hits[layer])[0] for layer, c in candidates.items()}
                    selections = {layer: rows for layer, rows in selections.items() if rows}
                    prefetched = manager.read(selections)
                    for layer, rows in selections.items():
                        candidates[layer]._prefetched = (rows, prefetched[layer])
                    for layer, reference in references.items():
                        candidate = candidates[layer]
                        assert reference.promote(hits[layer]) == candidate.promote(hits[layer])
                        assert reference.last_floor_deferred == candidate.last_floor_deferred
                        count = len(reference.sram_pos)
                        scores = torch.arange(count, device='cuda', dtype=torch.float32) % 7
                        for cache in (reference, candidate):
                            cache.sram_cum.copy_(scores)
                            cache.sram_age.fill_(step + 1)
                            cache._t = step + 1
                        keys, values = (torch.randn(8, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2))
                        assert reference.evict_and_demote(keys, values, 256 + step) == candidate.evict_and_demote(keys, values, 256 + step)
                    # Flush shared writes once, then check complete state and charged payload.
                    manager.flush()
                    for layer, reference in references.items():
                        check_state(reference, candidates[layer])
                        assert len(set(candidates[layer]._logical + candidates[layer]._free)) == candidates[layer].capacity
                # Explicit repeated-slot reuse before a flush must retain submitted order.
                cache = candidates[0]
                keep = list(range(len(cache._logical) - 1))
                cache.compact_stt_pair(keep)
                a, b = (torch.randn(8, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2))
                cache.append_stt_pair(a, a)
                reused = cache._logical[-1]
                cache.compact_stt_pair(list(range(len(cache._logical) - 1)))
                cache._free.remove(reused)
                cache._free.insert(0, reused)
                calls = manager.d2h_calls
                cache.append_stt_pair(b, b)
                assert manager.d2h_calls == calls + 1, 'overlapping pending slots did not flush'
                manager.synchronize()
                assert torch.equal(cache.pool.reshape(2, 8, cache.capacity, 128)[:, :, reused], torch.stack((b[:, 0], b[:, 0])))
                before = list(cache._logical)
                try:
                    huge = torch.empty(8, cache.capacity + 1, 128, device='cuda', dtype=torch.bfloat16)
                    cache.append_stt_pair(huge, huge)
                except ValueError:
                    pass
                else:
                    raise AssertionError('capacity guard missing')
                assert before == cache._logical
                # Reset owns a queued write as well as previously dispatched transfers.
                if not cache._free:
                    cache.compact_stt_pair(list(range(len(cache._logical) - 1)))
                cache.append_stt_pair(a, b)
                assert manager.pending
                policy = CpuBatchPolicy(sram_budget=32, stt_budget=64)
                policy.layer_caches, policy.manager = candidates, manager
                policy.reset()
                assert manager.pool is None and not policy.layer_caches
                assert all(c.pool is None and c.manager is None for c in candidates.values())
                report['checks'].append(dict(inclusive=inclusive, initial_fill=fill, steps=24, layers=2,
                    exact_state_payload=True, duplicate_slot_write_order=True, capacity_guard=True, queued_write_reset=True))
                manager, candidates = None, {}
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if manager is not None:
            manager.close()
        for cache in candidates.values():
            cache.close()
        atomic_json(args.output, report)
    print(json.dumps(dict(passed=report['passed'], checks=len(report['checks']))))


if __name__ == '__main__':
    main()
