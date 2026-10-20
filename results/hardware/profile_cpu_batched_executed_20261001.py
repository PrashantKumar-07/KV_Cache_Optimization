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

from compare_cpu_batched import BatchPolicy
from latency_profile import profile_one
from provenance import atomic_json, runtime_identity


class ProfilePolicy(BatchPolicy):
    def __init__(self, **kwargs):
        self.profiler = cProfile.Profile()
        super().__init__(**kwargs)

    def _decode_step(self, past):
        self.profiler.enable()
        try:
            return super()._decode_step(past)
        finally:
            self.profiler.disable()

    def update_scores(self, *args, **kwargs):
        self.profiler.enable()
        try:
            return super().update_scores(*args, **kwargs)
        finally:
            self.profiler.disable()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile_path, text_path = args.output.with_suffix('.prof'), args.output.with_suffix('.prof.txt')
    if any(path.exists() for path in (args.output, profile_path, text_path)):
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = dict(passed=False, scope=__doc__, runtime=runtime_identity(),
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ('experiments/profile_cpu_batched.py', 'experiments/cpu_batched_cache.py', 'experiments/cpu_slot_cache.py', 'experiments/compare_cpu_batched.py')})
    policy = None
    try:
        prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
        stream = json.loads(Path('results/shadow_ablation_full/fixed_stream_20260929.json').read_text())['config']['teacher_token_ids']
        report.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), teacher_token_ids=stream[:21], seed=0, context=16384, decode_steps=20, gpu=torch.cuda.get_device_name())
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct', torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        policy = ProfilePolicy(sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
                               kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
        report['profiled_run_not_latency_gate'] = profile_one(model, tokenizer, policy, 16384,
            n_decode=20, n_warmup=0, n_repeat=1, device='cuda', prompt_text=prompt, forced_token_ids=stream)
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
        if policy is not None:
            policy.reset()
        atomic_json(args.output, report)
    print(text.getvalue(), flush=True)


if __name__ == '__main__':
    main()
