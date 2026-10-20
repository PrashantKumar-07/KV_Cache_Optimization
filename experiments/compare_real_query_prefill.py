"""Same-process ABBA prefill/placement cost for K-proxy vs actual queries."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import longbench_eval as bench
from evaluate_query_selection import PromptQueryCapture
from provenance import atomic_json, runtime_identity


def input_ids(tokenizer, data_dir, task, device):
    path = bench.task_data_path(data_dir, task)
    item = json.loads(path.open(encoding='utf-8').readline())
    prompt = bench.DATASET_TO_PROMPT[task].format(**item)
    if task not in bench.NO_CHAT_TASKS:
        prompt = f'[INST] {prompt} [/INST]'
    ids = tokenizer(prompt, return_tensors='pt').input_ids
    if ids.size(1) > 31500:
        ids = torch.cat((ids[:, :15750], ids[:, -15750:]), dim=1)
    return ids.to(device), path


def measure(model, ids, real_query):
    policy = bench.TieredKVPolicy(sram_budget=992, stt_budget=3104,
        start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
        sttram_bifurcation_frac=.75, stt_expose_quota=32)
    policy.prompt_query_chunk = 128
    capture = PromptQueryCapture(model, policy) if real_query else None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.inference_mode(), bench._attn_impl(model, 'sdpa'):
            out = model(ids, use_cache=True,
                        position_ids=torch.arange(ids.size(1), device=ids.device)[None])
            policy(bench.to_tuple_kv(out.past_key_values))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        positions = {str(i): {'fast': c.sram_pos, 'cold': c.stt_pos}
                     for i, c in policy.layer_caches.items()}
        placement_sha = hashlib.sha256(json.dumps(positions, sort_keys=True).encode()).hexdigest()
        return {'seconds': seconds, 'peak_alloc_bytes': peak,
                'placement_sha256': placement_sha}
    finally:
        if capture:
            capture.close()
        del policy
        if 'out' in locals():
            del out
        gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='../models/mistral-7b-instruct')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--tasks', nargs='+', default=['qasper', 'narrativeqa'])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report = {'runtime': runtime_identity(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model': args.model, 'tasks': {},
              'protocol': 'GPU0, bf16, first task example, SDPA full prompt; two warmups then '
                          'three ABBA blocks. Time includes model prefill and cache placement, '
                          'with CUDA completion; peak allocated includes model weights.'}
    for task in args.tasks:
        ids, source = input_ids(tokenizer, args.data_dir, task, 'cuda')
        rows = []
        for real in (False, True):
            measure(model, ids, real)
        for block in range(3):
            for real in (False, True, True, False):
                rows.append({'block': block, 'arm': 'actual_query' if real else 'proxy',
                             **measure(model, ids, real)})
        for arm in ('proxy', 'actual_query'):
            hashes = {r['placement_sha256'] for r in rows if r['arm'] == arm}
            assert len(hashes) == 1, (task, arm, hashes)
        report['tasks'][task] = {
            'prompt_tokens': ids.size(1),
            'input_ids_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'rows': rows}
        print(task, 'done', flush=True)
    atomic_json(args.output, report)


if __name__ == '__main__':
    main()
