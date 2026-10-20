"""Diagnostic policy-only cProfile; profiled timings are not acceptance evidence."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compare_cpu_batched import BatchPolicy, DirectPolicy
from evaluate_query_selection import PromptQueryCapture
from latency_profile import profile_one
from provenance import atomic_json, runtime_identity


def nvtx_call(label, function):
    def wrapped(*args, **kwargs):
        with torch.cuda.nvtx.range(label):
            return function(*args, **kwargs)
    return wrapped


class ProfileCalls:
    def __init__(self, **kwargs):
        self.profiler = cProfile.Profile()
        super().__init__(**kwargs)

    def _bifurcate(self, past, **prefill_state):
        result = super()._bifurcate(past, **prefill_state)
        if getattr(self, 'trace_cuda', False):
            for layer, cache in self.layer_caches.items():
                for name in ('evict_and_demote','_select_hot_victims','promote'):
                    setattr(cache, name, nvtx_call(f'layer{layer}/{name}',getattr(cache,name)))
            for name in ('_prepare_decode_reads','_visible_tuple'):
                setattr(self,name,nvtx_call(name,getattr(self,name)))
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            self.trace_started = True
        return result

    def _decode_step(self, past):
        self.profiler.enable()
        try:
            result = super()._decode_step(past)
            self.geometry.append(dict(step=self.decode_step_count,
                cold_min=min(len(c.stt_pos) for c in self.layer_caches.values()),
                cold_max=max(len(c.stt_pos) for c in self.layer_caches.values()),
                saturated_layers=sum(len(c.stt_pos)==c.cfg.sttram_capacity for c in self.layer_caches.values())))
            return result
        finally:
            self.profiler.disable()

    def update_scores(self, *args, **kwargs):
        self.profiler.enable()
        try:
            return super().update_scores(*args, **kwargs)
        finally:
            self.profiler.disable()


class ProfilePolicy(ProfileCalls, BatchPolicy):
    pass


class ProfileDirectPolicy(ProfileCalls, DirectPolicy):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direct', action='store_true', help='Current direct CPU/score batching path')
    parser.add_argument('--real-query-prefill', action='store_true')
    parser.add_argument('--decode', type=int, default=20)
    parser.add_argument('--trace-cuda', action='store_true',help='NVTX and CUDA profiler API collection after prefill')
    args = parser.parse_args()
    if args.decode < 1: parser.error('--decode must be positive')
    profile_path, text_path = args.output.with_suffix('.prof'), args.output.with_suffix('.prof.txt')
    if any(path.exists() for path in (args.output, profile_path, text_path)):
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = dict(passed=False, direct=args.direct, real_query_prefill=args.real_query_prefill, trace_cuda=args.trace_cuda, scope=__doc__, runtime=runtime_identity(),
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ('experiments/profile_cpu_batched.py', 'experiments/cpu_batched_cache.py', 'experiments/cpu_slot_cache.py', 'experiments/compare_cpu_batched.py', 'experiments/direct_shared_pool.py', 'experiments/batched_score_policy.py', 'experiments/evaluate_query_selection.py')})
    policy = None
    capture = None
    try:
        prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
        stream = json.loads(Path('results/shadow_ablation_full/fixed_stream_20260929.json').read_text())['config']['teacher_token_ids']
        if len(stream)<args.decode+1:raise ValueError('Fixed teacher stream is too short')
        report.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), teacher_token_ids=stream[:args.decode+1], seed=0, context=16384, decode_steps=args.decode, gpu=torch.cuda.get_device_name())
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct', torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        policy = (ProfileDirectPolicy if args.direct else ProfilePolicy)(sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
                               kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
        policy.geometry = []
        policy.trace_cuda = args.trace_cuda
        policy.trace_started = False
        if args.real_query_prefill:
            policy.prompt_query_score_mode = 'chunked'
            policy.prompt_query_chunk = 128
            capture = PromptQueryCapture(model, policy)
        report['profiled_run_not_latency_gate'] = profile_one(model, tokenizer, policy, 16384,
            n_decode=args.decode, n_warmup=0, n_repeat=1, device='cuda', prompt_text=prompt, forced_token_ids=stream)
        if policy.trace_started:
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
            policy.trace_started = False
        report['cold_geometry'] = policy.geometry
        args.output.parent.mkdir(parents=True, exist_ok=True)
        policy.profiler.dump_stats(str(profile_path))
        text = io.StringIO()
        pstats.Stats(policy.profiler, stream=text).strip_dirs().sort_stats('cumulative').print_stats(60)
        text_path.write_text(text.getvalue())
        report.update(passed=True, profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                      profile_text_sha256=hashlib.sha256(text_path.read_bytes()).hexdigest(),
                      limitations='Only decode/score policy calls profiled. Implicit GPU waits and instrumentation distort times; overlapping call times are not additive savings. No speed gate or quality claim.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if policy is not None and getattr(policy,'trace_started',False):
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
        if capture is not None:
            capture.close()
        if policy is not None:
            policy.reset()
        atomic_json(args.output, report)
    print(text.getvalue(), flush=True)


if __name__ == '__main__':
    main()
