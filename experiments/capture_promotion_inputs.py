"""Record actual ordinary-policy promotion inputs; observer timing is invalid."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import torch
import profile_native_reuse as driver
from cpu_batched_cache import CpuBatchPolicy
from provenance import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    target = args.directory / 'inputs.pt'
    if target.exists(): raise FileExistsError(target)
    frames, current = [], []
    original_prepare, original_prefetch = CpuBatchPolicy._prepare_decode_reads, CpuBatchPolicy._prefetch

    def prepare(policy):
        frame = dict(step=policy.decode_step_count, budget=policy.promotion_budget,
            attention=copy.deepcopy(policy._stt_attn), attention_indices=copy.deepcopy(policy._stt_attn_idx), layers={})
        for layer, cache in policy.layer_caches.items():
            names = ('stt_shadow', 'stt_was_vram_resident', 'stt_peek_streak')
            assert all(getattr(cache, n).device.type == 'cpu' for n in names)
            frame['layers'][layer] = dict(positions=list(cache.stt_pos),
                inclusive=cache.cfg.inclusive, floor=cache.cfg.stt_live_floor,
                hysteresis=cache._HYSTERESIS_MIN_STREAK,
                **{n: getattr(cache, n).clone() for n in names})
        current.append(frame)
        try:
            result = original_prepare(policy)
            frame['expected_plans'] = copy.deepcopy(policy._planned_promotions)
            frame['expected_streaks'] = {l: c.stt_peek_streak.clone() for l, c in policy.layer_caches.items()}
            frames.append(frame)
            return result
        finally:
            current.pop()

    def prefetch(policy, selections):
        if current: current[-1]['expected_selections'] = copy.deepcopy(selections)
        return original_prefetch(policy, selections)

    CpuBatchPolicy._prepare_decode_reads, CpuBatchPolicy._prefetch = prepare, prefetch
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], '--output', str(args.directory / 'observed_profile.json'),
                    '--direct', '--real-query-prefill', '--decode', '10']
        driver.main()
        assert len(frames) == 10 and all(len(f['layers']) == 32 for f in frames)
        torch.save(frames, target)
        atomic_json(args.directory / 'capture_report.json', dict(passed=True, frames=10,
            layer_inputs=320, input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations='CPU metadata snapshots and expected decisions only. Observer invalidates profiled timings. No KV/model speed or quality claim. Fixed inspected development stream.'))
    finally:
        sys.argv = saved_argv
        CpuBatchPolicy._prepare_decode_reads, CpuBatchPolicy._prefetch = original_prepare, original_prefetch


if __name__ == '__main__': main()
