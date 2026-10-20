"""Line-level host timing of current 16K TieredKV policy hot functions."""
from collections import defaultdict
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_query_selection import PromptQueryCapture
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity
from tiered_kv_cache import TieredKVCache


def main():
    model_path = '../models/mistral-7b-instruct'
    prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = TieredKVPolicy(sram_budget=992, stt_budget=3104,
        stt_expose_quota=32, start_size=4, recent_size=128,
        kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
    capture = PromptQueryCapture(model, policy)
    targets = {TieredKVCache.evict_and_demote.__code__,
               TieredKVPolicy._visible_tuple.__code__}
    totals = defaultdict(int)
    counts = defaultdict(int)

    def global_trace(frame, event, arg):
        if event != 'call' or frame.f_code not in targets:
            return None
        name = frame.f_code.co_name
        counts[name] += 1
        previous_line = None
        previous_at = time.perf_counter_ns()

        def local_trace(frame, event, arg):
            nonlocal previous_line, previous_at
            now = time.perf_counter_ns()
            if previous_line is not None:
                totals[(name, previous_line)] += now - previous_at
            previous_line = frame.f_lineno if event == 'line' else None
            previous_at = now
            return local_trace
        return local_trace

    try:
        sys.settrace(global_trace)
        result = profile_one(model, tokenizer, policy, 16384,
            n_decode=8, n_warmup=0, n_repeat=1, prompt_text=prompt)
    finally:
        sys.settrace(None)
        capture.close()
    source = {code.co_name: Path(code.co_filename).read_text().splitlines()
              for code in targets}
    rows = sorted(({'function': name, 'line': line,
                    'host_ms': ns / 1e6,
                    'source': source[name][line - 1].strip()}
                   for (name, line), ns in totals.items()),
                  key=lambda row: row['host_ms'], reverse=True)
    artifact = {'config': {'runtime': runtime_identity(), 'context': 16384,
        'decode': 8, 'prompt_file': 'results/remediation_v3/latency_prompt.txt',
        'scope': 'host line trace includes nested calls and GPU waits'},
        'observed_itl_ms': result['itl_ms'], 'observed_forward_ms': result['forward_ms'],
        'calls': dict(counts), 'rows': rows}
    path = 'results/baseline_fidelity/policy_line_16k_8.json'
    atomic_json(path, artifact)
    for row in rows[:15]:
        print(row['function'], row['line'], round(row['host_ms'], 2), row['source'])


if __name__ == '__main__':
    main()
