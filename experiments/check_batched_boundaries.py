"""Exact-ID boundary/retrieval stress; oracle quality is descriptive."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from check_batched_generation import CapturedTokenizer, GpuStress, CpuStress
from compare_cpu_batched import digest
import longbench_eval as bench
from provenance import atomic_json, runtime_identity


class FixedTokenizer(CapturedTokenizer):
    def __init__(self, tokenizer, ids):
        super().__init__(tokenizer)
        self.input_ids = torch.tensor(ids, dtype=torch.long).view(1, -1)

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=self.input_ids.clone())


class Observe:
    def __init__(self, **kwargs):
        self.answer_span = None
        self.initial_answer_retention = {}
        self.initial_cache_rows = {}
        self.prefill_query_shapes = {}
        self.max_cold_rows = {}
        self.largest_promoted = self.largest_paid_writes = 0
        self.previous_promoted = self.previous_paid = 0
        super().__init__(**kwargs)

    def _visible_tuple(self, past):
        result = super()._visible_tuple(past)
        for layer, cache in self.layer_caches.items():
            self.max_cold_rows[layer] = max(self.max_cold_rows.get(layer, 0), len(cache.stt_pos))
            if self.frames == 1 and self.answer_span is not None:
                span = set(range(*self.answer_span))
                hot, cold = set(cache.sram_pos), set(cache.stt_pos)
                self.initial_cache_rows[layer] = dict(hot_rows=len(hot), cold_rows=len(cold),
                    hot_payload_bytes=sum(t.numel()*t.element_size() for t in (cache.sram_k, cache.sram_v)),
                    cold_payload_bytes=len(cold)*cache.cfg.num_heads*cache.cfg.head_dim*2*torch.empty((), dtype=cache.cfg.resolved_stt_kv_dtype).element_size(),
                    visible_rows=result[layer][0].shape[-2])
                self.initial_answer_retention[layer] = dict(span_tokens=len(span),
                    hot_tokens=len(span & hot), cold_tokens=len(span & cold),
                    retained_tokens=len(span & (hot | cold)))
        totals = self._live_totals()
        promoted, paid = totals['promoted'], totals['paid_writes']
        self.largest_promoted = max(self.largest_promoted, promoted - self.previous_promoted)
        self.largest_paid_writes = max(self.largest_paid_writes, paid - self.previous_paid)
        self.previous_promoted, self.previous_paid = promoted, paid
        return result


class GpuObserved(Observe, GpuStress):
    pass


class CpuObserved(Observe, CpuStress):
    pass


def measure(model, tokenizer, case, arm, *, prompt_queries=False):
    if prompt_queries and arm not in ('gpu', 'cpu_scores'):
        raise ValueError('Query capture requires a Tiered GPU or batched CPU policy')
    policy = bench.FullCachePolicy() if arm == 'oracle' else (GpuObserved if arm == 'gpu' else CpuObserved)(
        sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
        kv_dtype=torch.bfloat16, sttram_bifurcation_frac=case['fill'])
    if arm != 'oracle':
        policy.answer_span = case.get('answer_span')
    capture = FixedTokenizer(tokenizer, case['ids'])
    query_capture = None
    try:
        if prompt_queries:
            from evaluate_query_selection import PromptQueryCapture
            policy.prompt_query_score_mode = 'chunked'
            policy.prompt_query_chunk = 128
            query_capture = PromptQueryCapture(model, policy)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prediction = bench.generate_with_policy(model, capture, '', case['max_gen'], policy, 'cuda', max_ctx=31500)
        torch.cuda.synchronize()
        ids = capture.ids or []
        if case['max_gen']:
            assert capture.ids is not None
        result = dict(prediction=prediction, generated_token_ids=ids,
            generation=getattr(policy, 'last_generation', {}),
            answer_correct=None if case['answer'] is None else case['answer'].casefold() in prediction.casefold(),
            elapsed_s_diagnostic=time.perf_counter() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            final_gpu_allocated_bytes=torch.cuda.memory_allocated())
        if arm != 'oracle':
            caches = list(policy.layer_caches.values())
            result.update(state_sha256=digest(policy.layer_caches),
                trajectory_sha256=policy.trajectory.hexdigest(), trajectory_frames=policy.frames,
                counters=policy._live_totals(), initial_answer_retention=policy.initial_answer_retention,
                initial_cache_rows=policy.initial_cache_rows, prefill_query_shapes=policy.prefill_query_shapes,
                max_cold_rows_per_layer=policy.max_cold_rows, largest_step_promoted=policy.largest_promoted,
                largest_step_paid_writes=policy.largest_paid_writes,
                explicit_h2d_payload_bytes=sum(getattr(c, 'host_read_bytes', 0) for c in caches),
                explicit_d2h_payload_bytes=sum(getattr(c, 'host_write_bytes', 0) for c in caches),
                occupied_host_tensor_bytes=sum(getattr(c, 'occupied_host_bytes', 0) for c in caches),
                reserved_host_tensor_bytes=sum(getattr(c, 'reserved_host_bytes', 0) for c in caches))
        return result
    finally:
        if query_capture is not None:
            query_capture.close()
        policy.reset()
        del policy
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = dict(passed=False, scope=__doc__, pid=os.getpid(), runtime=runtime_identity(), rows=[],
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('experiments/check_batched_boundaries.py', 'experiments/check_batched_generation.py',
             'experiments/batched_score_policy.py', 'experiments/cpu_batched_cache.py',
             'experiments/cpu_slot_cache.py', 'experiments/compare_cpu_batched.py')})
    try:
        manifest = json.loads(args.manifest.read_text())
        assert manifest['seed'] == 0 and manifest['max_ctx'] == 31500 and len(manifest['cases']) == 16
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'])
        report['gpu'] = torch.cuda.get_device_name()
        for number, case in enumerate(manifest['cases']):
            assert len(case['ids']) <= 31500
            assert hashlib.sha256(json.dumps(case['ids'], separators=(',', ':')).encode()).hexdigest() == case['token_sha256']
            reference = None
            order = ('gpu', 'cpu_scores') if number % 2 == 0 else ('cpu_scores', 'gpu')
            for arm in (*order, *(('oracle',) if case.get('oracle') else ())):
                result = measure(model, tokenizer, case, arm)
                report['rows'].append(dict(case=case['name'], prompt_tokens=len(case['ids']), arm=arm, **result))
                atomic_json(args.output, report)
                if arm != 'oracle':
                    if reference is None:
                        reference = result
                    else:
                        for field in ('prediction', 'generated_token_ids', 'generation', 'answer_correct',
                                      'state_sha256', 'trajectory_sha256', 'trajectory_frames', 'counters',
                                      'initial_answer_retention', 'max_cold_rows_per_layer',
                                      'largest_step_promoted', 'largest_step_paid_writes'):
                            assert result[field] == reference[field], f'{case["name"]}/{field} mismatch'
                print(case['name'], arm, len(result['generated_token_ids']), result['answer_correct'], flush=True)
        assert len(report['rows']) == 38
        report.update(passed=True, exact_boundary_and_retrieval_integrity=True,
            limitations='Synthetic repeated distractors and exact-ID prompts. Oracle is a descriptive full-cache reference, with unequal resources. Shared sparse-policy answer failures are not correctness of retrieval or quality superiority. Diagnostic times are not ABBA latency. No second model/concurrency/process-peak/physical STT evidence.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print('16 sparse pairs exact; oracle outcomes preserved', flush=True)


if __name__ == '__main__':
    main()
