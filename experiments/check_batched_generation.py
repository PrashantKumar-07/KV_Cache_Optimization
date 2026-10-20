"""Frozen free-generation QA/summarization integrity stress, not a speed benchmark."""
import argparse
from array import array
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from batched_score_policy import ScoreBatchPolicy
from compare_cpu_batched import digest
import longbench_eval as bench
from provenance import atomic_json, read_jsonl, runtime_identity


class CapturedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer, self.ids = tokenizer, None

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def decode(self, ids, *args, **kwargs):
        self.ids = list(ids)
        return self.tokenizer.decode(ids, *args, **kwargs)


class Trajectory:
    def _bifurcate(self, past, **prefill_state):
        self.prefill_query_shapes = {layer: list(q.shape)
                                    for layer, q in self.prompt_q_by_layer.items()}
        result = super()._bifurcate(past, **prefill_state)
        assert not self.prompt_q_by_layer, 'Placement left captured queries unconsumed'
        return result

    def _visible_tuple(self, past):
        result = super()._visible_tuple(past)
        for layer, cache in sorted(self.layer_caches.items()):
            selected = self._vis_stt_idx.get(layer, [])
            self.trajectory.update(array('q', [layer, len(cache.sram_pos), len(cache.stt_pos), len(selected)]).tobytes())
            for values in (cache.sram_pos, cache.stt_pos, selected):
                self.trajectory.update(array('q', values).tobytes())
            self.trajectory.update(cache.stt_shadow.cpu().numpy().tobytes())
        self.frames += 1
        return result


class GpuStress(Trajectory, bench.TieredKVPolicy):
    def __init__(self, **kwargs):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**kwargs)


class CpuStress(Trajectory, ScoreBatchPolicy):
    def __init__(self, **kwargs):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**kwargs)


def measure(model, tokenizer, item, task, seed, arm, max_ctx, *, prompt_queries=False,
            policy_cls=None, policy_options=None):
    torch.manual_seed(seed)
    options = dict(sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
                   kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
    options.update(policy_options or {})
    policy = (policy_cls or (GpuStress if arm == 'gpu' else CpuStress))(**options)
    capture = CapturedTokenizer(tokenizer)
    prompt = bench.DATASET_TO_PROMPT[task].format(**item)
    if task not in bench.NO_CHAT_TASKS:
        prompt = f'[INST] {prompt} [/INST]'
    maximum = bench.DATASET_TO_MAXGEN[task]
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
        prediction = bench.generate_with_policy(model, capture, prompt, maximum, policy, 'cuda', max_ctx=max_ctx, task_name=task)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert capture.ids is not None, 'generated IDs were not captured'
        final_gpu, peak_gpu = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
        score = 100 * max(bench.DATASET_TO_METRIC[task](prediction, answer, all_classes=item.get('all_classes', [])) for answer in item['answers'])
        caches = list(policy.layer_caches.values())
        result = dict(prediction=prediction, generated_token_ids=capture.ids, score=score,
            generation=policy.last_generation, max_gen=maximum,
            termination='eos' if capture.ids[-1] == tokenizer.eos_token_id else 'cap',
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            state_sha256=digest(policy.layer_caches), trajectory_sha256=policy.trajectory.hexdigest(),
            trajectory_frames=policy.frames, counters=policy._live_totals(),
            prefill_query_shapes=policy.prefill_query_shapes,
            elapsed_s_diagnostic=elapsed, final_gpu_allocated_bytes=final_gpu, peak_gpu_allocated_bytes=peak_gpu,
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
            ('experiments/check_batched_generation.py', 'experiments/batched_score_policy.py', 'experiments/cpu_batched_cache.py', 'experiments/cpu_slot_cache.py', 'experiments/compare_cpu_batched.py')},
        trajectory_encoding=dict(byteorder=sys.byteorder, index_bytes=array('q').itemsize))
    try:
        manifest = json.loads(args.manifest.read_text())
        requests = []
        for task, entry in manifest['tasks'].items():
            path = Path(entry['file'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
            data = read_jsonl(path)
            for selection in entry['selected']:
                item = data[selection['index']]
                assert hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest() == selection['sample_id']
                assert hashlib.sha256(item['context'].encode()).hexdigest() == selection['context_sha256']
                requests.append((task, selection, item))
        assert len(requests) == 12 and len({x[1]['context_sha256'] for x in requests}) == 12
        report.update(gpu=torch.cuda.get_device_name(), expected_pairs=24)
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'])
        for seed in manifest['seeds']:
            for number, (task, selection, item) in enumerate(requests):
                reference = None
                order = ('gpu', 'cpu_scores') if (number + seed) % 2 == 0 else ('cpu_scores', 'gpu')
                for arm in order:
                    result = measure(model, tokenizer, item, task, seed, arm, manifest['max_ctx'])
                    report['rows'].append(dict(task=task, sample_id=selection['sample_id'], context_sha256=selection['context_sha256'], seed=seed, arm=arm, **result))
                    atomic_json(args.output, report)
                    if reference is None:
                        reference = result
                    else:
                        for field in ('prediction', 'generated_token_ids', 'score', 'generation', 'max_gen', 'termination', 'prompt_sha256', 'state_sha256', 'trajectory_sha256', 'trajectory_frames', 'counters'):
                            assert result[field] == reference[field], f'{task}/{seed}/{selection["sample_id"]}/{field} mismatch'
                    print(task, number, seed, arm, len(result['generated_token_ids']), result['score'], flush=True)
        assert len(report['rows']) == 48
        report.update(passed=True, exact_free_generation_state_trajectory_and_quality=True,
            task_balanced_quality_delta=0., paired_quality_ci95=[0., 0.], quality_noninferiority_passed=True,
            limitations='Previously frozen quality slices reused, not unseen data. One model, greedy inference, 12 documents/two seeds. Diagnostic times include identical trajectory observers; no ABBA speed claim. No concurrency, full boundary/needle suite, allocator/process peak or physical STT/controller evidence.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print('all 24 free-generation pairs exact', flush=True)


if __name__ == '__main__':
    main()
