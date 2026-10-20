"""Charged actual-policy versus batched attention bookkeeping falsification screen."""
import argparse
import hashlib
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from batched_score_policy import batched_update
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity



def fixture(weights, cumulative, cold, changed=False):
    policy = TieredKVPolicy()
    policy.layer_caches, policy._vis_sram_keep, policy._vis_stt_pos, policy._vis_stt_idx = {}, {}, {}, {}
    for layer in range(len(weights)):
        positions = [10000 + layer * 100 + i for i in range(64)]
        indices = [(layer + i * 3) % 64 for i in range(cold)]
        policy.layer_caches[layer] = SimpleNamespace(sram_cum=cumulative[layer].clone(),
            stt_pos=positions, stt_last_access=torch.zeros(64, dtype=torch.long), _t=17)
        policy._vis_sram_keep[layer] = list(range(len(cumulative[layer])))
        policy._vis_stt_idx[layer] = indices
        policy._vis_stt_pos[layer] = [positions[i] for i in indices]
    if changed:
        policy._vis_stt_pos[0] = list(reversed(policy._vis_stt_pos[0]))
        policy._vis_stt_idx[1] = None
    return policy


def assert_equal(left, right):
    assert left._stt_attn == right._stt_attn and left._stt_attn_idx == right._stt_attn_idx
    for layer in left.layer_caches:
        a, b = left.layer_caches[layer], right.layer_caches[layer]
        assert torch.equal(a.sram_cum, b.sram_cum), f'layer {layer}: FP32 score mismatch'
        assert torch.equal(a.stt_last_access, b.stt_last_access), f'layer {layer}: LRU mismatch'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = dict(passed=False, runtime=runtime_identity(), scope=__doc__, rows=[], checks=[],
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  backend_sha256=hashlib.sha256((Path(__file__).parent / 'batched_score_policy.py').read_bytes()).hexdigest())
    try:
        torch.manual_seed(0)
        report.update(gpu=torch.cuda.get_device_name(), seed=0, shape=[32, 32, 1057], warmup=4)
        cumulative = torch.rand(32, 1024, device='cuda')
        past = [(None, None)] * 32
        for label, dtype, cold, changed in [('bf16', torch.bfloat16, 32, False),
                ('fp32', torch.float32, 32, False), ('zero', torch.bfloat16, 0, False),
                ('mapping', torch.bfloat16, 32, True)]:
            weights = [torch.rand((1, 32, 1, 1024 + cold + 1), dtype=dtype, device='cuda') for _ in range(32)]
            for weight in weights:
                weight[:, :, :, 1024:1028] = 0
            reference = fixture(weights, cumulative, cold, changed)
            candidate = fixture(weights, cumulative, cold, changed)
            for _ in range(5):
                TieredKVPolicy.update_scores(reference, past, weights)
                batched_update(candidate, past, weights)
                assert_equal(reference, candidate)
            report['checks'].append(dict(regime=label, exact_scores_dictionaries_lru=True))
            if label != 'bf16':
                continue
            for block in range(3):
                reference = None
                for kind in ('baseline', 'batched', 'batched', 'baseline'):
                    policy = fixture(weights, cumulative, cold)
                    samples = []
                    for step in range(20):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        (TieredKVPolicy.update_scores if kind == 'baseline' else batched_update)(policy, past, weights)
                        torch.cuda.synchronize()
                        samples.append(1000 * (time.perf_counter() - start))
                        if reference is None:
                            reference = fixture(weights, cumulative, cold)
                        TieredKVPolicy.update_scores(reference, past, weights)
                        assert_equal(reference, policy)
                    reference = None
                    report['rows'].append(dict(block=block, arm=kind, warmup_ms=samples[:4], step_ms=samples[4:]))
                    print(block, kind, round(statistics.median(samples[4:]), 3), flush=True)
        # Heterogeneous widths must take the unchanged baseline fallback.
        weights[0] = torch.rand((1, 32, 1, 1058), dtype=torch.bfloat16, device='cuda')
        reference, candidate = (fixture(weights, cumulative, 32) for _ in range(2))
        TieredKVPolicy.update_scores(reference, past, weights)
        batched_update(candidate, past, weights)
        assert_equal(reference, candidate)
        report['checks'].append(dict(regime='heterogeneous_fallback', exact_scores_dictionaries_lru=True))
        summary, blocks = {}, []
        for kind in ('baseline', 'batched'):
            samples = sorted(t for row in report['rows'] if row['arm'] == kind for t in row['step_ms'])
            summary[kind] = dict(median_ms=statistics.median(samples), p95_ms=samples[int(.95 * (len(samples) - 1))])
        for block in range(3):
            blocks.append({kind: statistics.median(t for row in report['rows'] if row['block'] == block and row['arm'] == kind for t in row['step_ms']) for kind in summary})
        saving = summary['baseline']['median_ms'] - summary['batched']['median_ms']
        report.update(passed=True, summary=summary, blocks=blocks, median_saving_ms=saving,
                      advance_to_model=saving >= 4 and all(b['batched'] < b['baseline'] for b in blocks) and summary['batched']['p95_ms'] <= 1.02 * summary['baseline']['p95_ms'],
                      limitations='Synthetic attention bookkeeping; repeated fixed weights, no model/policy execution, quality or whole-loop speed claim. Packing/casts/guards/metadata and synchronization charged.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print({key: report[key] for key in ('summary', 'median_saving_ms', 'advance_to_model')})


if __name__ == '__main__':
    main()
