"""Fixed-stream real-cache parity and paired CPU-fast/native-slot model trials."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from check_cpu_slots import FIELDS
from cpu_slot_cache import CpuSlotPolicy
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from probe_cpu_cold import CpuColdIndexSelectCache
from provenance import atomic_json, runtime_identity


class Recording:
    def _visible_tuple(self, past):
        result = super()._visible_tuple(past)
        self.history.append({i: (tuple(c.sram_pos), tuple(c.stt_pos), tuple(self._vis_stt_idx.get(i, ())), tuple(c.stt_shadow.tolist())) for i, c in self.layer_caches.items()})
        return result


class FastPolicy(Recording, TieredKVPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_cls = CpuColdIndexSelectCache
        self.history = []


class SlotPolicy(Recording, CpuSlotPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []


def digest(caches):
    value = hashlib.sha256()
    for layer, cache in sorted(caches.items()):
        value.update(str(layer).encode())
        for field in FIELDS:
            item = getattr(cache, field)
            value.update(field.encode())
            if isinstance(item, torch.Tensor):
                value.update(str((item.dtype, tuple(item.shape))).encode())
                value.update(item.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            else:
                value.update(json.dumps(item).encode())
        value.update(json.dumps(cache.metrics.summary(), sort_keys=True).encode())
    return value.hexdigest()


def measure(model, tokenizer, prompt, stream, ctx, decode, mode, warmup):
    cls = FastPolicy if mode == 'fast' else SlotPolicy
    policy = cls(sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
                 kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
    try:
        result = profile_one(model, tokenizer, policy, ctx, n_decode=decode,
            n_warmup=warmup, n_repeat=1, device='cuda', prompt_text=prompt,
            forced_token_ids=stream)
        result['final_gpu_allocated_bytes'] = torch.cuda.memory_allocated()
        result['state_sha256'] = digest(policy.layer_caches)
        result['history_sha256'] = hashlib.sha256(json.dumps(policy.history, sort_keys=True).encode()).hexdigest()
        caches = list(policy.layer_caches.values())
        result['explicit_h2d_payload_bytes'] = sum(c.host_read_bytes for c in caches)
        result['explicit_d2h_payload_bytes'] = sum(c.host_write_bytes for c in caches)
        result['initial_d2h_payload_bytes'] = sum(c.host_initial_write_bytes for c in caches)
        if mode == 'slots':
            result['occupied_host_bytes'] = sum(c.occupied_host_bytes for c in caches)
            result['reserved_host_bytes'] = sum(c.reserved_host_bytes for c in caches)
            result['native_d2h_calls'] = sum(c.d2h_calls for c in caches)
            result['native_h2d_calls'] = sum(c.h2d_calls for c in caches)
        else:
            payload = sum(t.numel() * t.element_size() for c in caches for t in (c.stt_k, c.stt_v))
            result['occupied_host_bytes'] = result['reserved_host_bytes'] = payload
        return result
    finally:
        policy.reset()
        del policy
        gc.collect()
        torch.cuda.empty_cache()


def check(reference, candidate):
    fields = ('state_sha256', 'history_sha256', 'explicit_h2d_payload_bytes',
              'explicit_d2h_payload_bytes', 'initial_d2h_payload_bytes', 'occupied_host_bytes')
    for field in fields:
        assert reference[field] == candidate[field], f'{field} differs'
    for a, b in zip(reference['trials'], candidate['trials']):
        for field in ('predicted_token_ids', 'paid_writes', 'writes_saved', 'demoted', 'promoted', 'decode_steps'):
            assert a[field] == b[field], f'{field} differs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contexts', type=int, nargs='+', default=[4096])
    parser.add_argument('--decode', type=int, default=5)
    parser.add_argument('--blocks', type=int, default=0, help='0: parity pair only; 3: ABBA latency trial')
    parser.add_argument('--warmup', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    if args.blocks not in (0, 3):
        parser.error('use 0 for parity or 3 for preregistered ABBA')
    report = {'config': vars(args) | {'output': str(args.output)}, 'runtime': runtime_identity(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'backend_sha256': hashlib.sha256(Path('experiments/cpu_slot_cache.py').read_bytes()).hexdigest(),
              'scope': 'CPU-DRAM surrogate, fixed-stream full-loop model/policy; observed history overhead included',
              'rows': [], 'passed': False}
    try:
        prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
        stream = json.loads(Path('results/shadow_ablation_full/fixed_stream_20260929.json').read_text())['config']['teacher_token_ids']
        report.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), teacher_token_ids=stream[:args.decode + 1])
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct', torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        report['gpu'] = torch.cuda.get_device_name()
        for ctx in args.contexts:
            reference = None
            for block in range(args.blocks or 1):
                for arm in (('fast', 'slots', 'slots', 'fast') if args.blocks else ('fast', 'slots')):
                    result = measure(model, tokenizer, prompt, stream, ctx, args.decode, arm, args.warmup)
                    if reference is None:
                        reference = result
                    else:
                        check(reference, result)
                    report['rows'].append(dict(context=ctx, block=block, arm=arm, **result))
                    atomic_json(args.output, report)
                    print(ctx, block, arm, result['itl_ms'], 'exact', flush=True)
        report['exact_model_policy_state_and_payload'] = True
        if args.blocks:
            summaries = {}
            for ctx in args.contexts:
                summary = {}
                for arm in ('fast', 'slots'):
                    steps = [ms for row in report['rows'] if row['context'] == ctx and row['arm'] == arm for trial in row['trials'] for ms in trial['step_ms']]
                    summary[arm] = dict(median_ms=statistics.median(steps), p95_ms=float(np.percentile(steps, 95)))
                summary['latency_gate_passed'] = summary['slots']['median_ms'] <= .95 * summary['fast']['median_ms'] and summary['slots']['p95_ms'] <= 1.02 * summary['fast']['p95_ms']
                summaries[str(ctx)] = summary
            report['summary'] = summaries
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps({'passed': report['passed'], 'summary': report.get('summary')}), flush=True)


if __name__ == '__main__':
    main()
