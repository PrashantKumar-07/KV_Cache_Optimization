#!/usr/bin/env python3
"""End-to-end fixed-length decode timing, including cache policy processing.

Use --prompt-file for real text. Synthetic timing is explicitly labeled.
Historical latency_headline.json measured model.forward only and is not
compatible with this benchmark's schema or end-to-end claims.
"""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from longbench_eval import (FullCachePolicy, StreamingLLMPolicy, H2OPolicy,
    SnapKVPolicy, TieredKVPolicy, to_tuple_kv, apply_tuple_kv_inplace, _attn_impl)
from provenance import atomic_json, resume_results, runtime_identity


def cuda_sync_time(fn, device=None):
    use_cuda = torch.cuda.is_available() and (device is None or str(device).startswith('cuda'))
    if use_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if use_cuda:
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000, result


def make_policy(name, budget, sink=4, *, sttram_bifurcation_frac=.75,
                tiered_vram_budget=None, tiered_stt_budget=None):
    recent = max(16, budget // 8)
    if name == 'full':
        return FullCachePolicy()
    if name == 'streamingllm':
        return StreamingLLMPolicy(sink, budget - sink)
    if name == 'h2o':
        return H2OPolicy(budget, sink, recent)
    if name == 'snapkv':
        return SnapKVPolicy(budget, sink, recent)
    if name == 'tieredkv':
        return TieredKVPolicy(
            sram_budget=budget if tiered_vram_budget is None else tiered_vram_budget,
            stt_budget=2 * budget if tiered_stt_budget is None else tiered_stt_budget,
            start_size=sink, recent_size=recent, kv_dtype=torch.bfloat16,
            sttram_bifurcation_frac=sttram_bifurcation_frac)
    raise ValueError(name)


@torch.inference_mode()
def profile_one(model, tokenizer, policy, ctx_len, n_decode=50, n_warmup=2,
                n_repeat=3, device='cuda', prompt_text=None, forced_token_ids=None):
    if min(ctx_len, n_decode, n_repeat) < 1 or n_warmup < 0:
        raise ValueError('Positive context/decode/repeat counts and nonnegative warmup required')
    if prompt_text is None:
        prompt_text = 'The quick brown fox jumps over the lazy dog . ' * (ctx_len + 1)
    ids = tokenizer(prompt_text, truncation=False, return_tensors='pt').input_ids
    if ids.size(1) < ctx_len:
        raise ValueError(f'Prompt has {ids.size(1)} tokens, fewer than requested {ctx_len}')
    ids = ids[:, :ctx_len].to(device)
    if forced_token_ids is not None and len(forced_token_ids) < n_decode + 1:
        raise ValueError('forced_token_ids needs prefill token plus one per decode forward')
    forced = (torch.tensor(forced_token_ids[:n_decode + 1], device=device).view(1, -1)
              if forced_token_ids is not None else None)
    use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()
    trials = []
    for trial in range(n_warmup + n_repeat):
        policy.reset()
        before_stats = policy.stats() if forced is not None else None
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        components = {}

        def prefill():
            pos = torch.arange(ctx_len, device=device).unsqueeze(0)
            with _attn_impl(model, 'sdpa'):
                model_ms, out = cuda_sync_time(
                    lambda: model(ids, use_cache=True, position_ids=pos), device)
            components['prefill_forward_ms'] = model_ms
            raw = out.past_key_values
            apply_tuple_kv_inplace(raw, policy(to_tuple_kv(raw)))
            token = out.logits[:, -1:].argmax(dim=-1)
            token.item()  # include host-visible token delivery
            if forced is not None:
                token = forced[:, :1]
            return raw, token

        ttft, (past, token) = cuda_sync_time(prefill, device)
        if use_cuda:
            prefill_peak = torch.cuda.max_memory_allocated(device) / 1e9
            torch.cuda.reset_peak_memory_stats(device)
        else:
            prefill_peak = 0.0
        steps, forwards = [], []
        predicted_tokens = []
        for step in range(n_decode):
            def decode():
                pos = torch.tensor([[ctx_len + step]], device=device)
                forward_ms, out = cuda_sync_time(lambda: model(
                    token, past_key_values=past, use_cache=True,
                    position_ids=pos, output_attentions=True), device)
                raw = out.past_key_values
                tup = to_tuple_kv(raw)
                if hasattr(policy, 'update_scores'):
                    policy.update_scores(tup, out.attentions)
                apply_tuple_kv_inplace(raw, policy(tup))
                nxt = out.logits[:, -1:].argmax(dim=-1)
                predicted = nxt.item()
                if forced is not None:
                    nxt = forced[:, step + 1:step + 2]
                return raw, nxt, forward_ms, predicted
            elapsed, (past, token, forward_ms, predicted) = cuda_sync_time(decode, device)
            steps.append(elapsed)
            forwards.append(forward_ms)
            predicted_tokens.append(predicted)
        peak = torch.cuda.max_memory_allocated(device) / 1e9 if use_cuda else 0.0
        after_stats = policy.stats() if forced is not None else None
        counters = ({k: after_stats[k] - before_stats[k]
                     for k in ('paid_writes', 'writes_saved', 'demoted',
                               'promoted', 'decode_steps')}
                    if forced is not None else {})
        if trial >= n_warmup:
            trials.append({'ttft_ms': ttft, 'step_ms': steps,
                           'forward_ms': forwards, 'prefill_peak_memory_gb': prefill_peak,
                           'decode_peak_memory_gb': peak,
                           'predicted_token_ids': predicted_tokens,
                           **components, **counters})
        del past, token
    return {
        'timing_scope': 'end_to_end_model_policy_token',
        'ttft_ms': round(float(np.median([x['ttft_ms'] for x in trials])), 3),
        'itl_ms': round(float(np.median([v for x in trials for v in x['step_ms']])), 3),
        'mean_itl_ms': float(np.mean([v for x in trials for v in x['step_ms']])),
        'forward_ms': float(np.median([v for x in trials for v in x['forward_ms']])),
        'peak_memory_gb': max(x['decode_peak_memory_gb'] for x in trials),
        'prefill_peak_memory_gb': max(x['prefill_peak_memory_gb'] for x in trials),
        'trials': trials,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', default='results/remediation/latency.json')
    p.add_argument('--ctx-lens', nargs='+', type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--tiered-vram-budget', type=int)
    p.add_argument('--tiered-stt-budget', type=int)
    p.add_argument('--sttram-bifurcation-frac', type=float, default=.75)
    p.add_argument('--n-decode', type=int, default=50)
    p.add_argument('--n-repeat', type=int, default=3)
    p.add_argument('--n-warmup', type=int, default=1)
    p.add_argument('--methods', nargs='+', default=['full','streamingllm','h2o','snapkv','tieredkv'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--prompt-file', help='UTF-8 real text; must contain enough tokens for every context')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    text = Path(args.prompt_file).read_text() if args.prompt_file else None
    config = {k:v for k,v in vars(args).items() if k != 'output'}
    config.update(runtime=runtime_identity(), timing_scope='end_to_end_model_policy_token',
                  prompt_kind='real_text' if text is not None else 'synthetic',
                  prompt_sha256=hashlib.sha256(text.encode()).hexdigest() if text else None,
                  generation='fixed_length_ignores_eos', schema_version=2)
    results = resume_results(args.output, config)
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map=args.device, attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for method in args.methods:
        results.setdefault(method, {})
        for ctx in args.ctx_lens:
            if str(ctx) in results[method]:
                continue
            policy = make_policy(method, args.budget,
                tiered_vram_budget=args.tiered_vram_budget,
                tiered_stt_budget=args.tiered_stt_budget,
                sttram_bifurcation_frac=args.sttram_bifurcation_frac)
            stats = profile_one(model, tokenizer, policy, ctx, args.n_decode,
                args.n_warmup, args.n_repeat, args.device, text)
            results[method][str(ctx)] = stats
            atomic_json(args.output, {'config':config,'results':results})
            print(f'{method} ctx={ctx} TTFT={stats["ttft_ms"]:.2f} ITL={stats["itl_ms"]:.2f} ms', flush=True)


if __name__ == '__main__':
    main()
