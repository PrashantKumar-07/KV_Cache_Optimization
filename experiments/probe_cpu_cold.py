"""Opt-in CPU-DRAM cold K/V probe; never used by the default TieredKV policy.

The host is a surrogate memory tier, not an STT-RAM device. This deliberately
uses simple synchronous pageable copies to establish correctness and a
measured lower bound on the cost of this existing per-layer cache layout.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity
from tiered_kv_cache import TieredKVCache


class CpuColdCache(TieredKVCache):
    """Store only cold K/V payload on CPU; retain current GPU policy metadata."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.host_read_bytes = 0
        self.host_write_bytes = 0
        self.host_initial_write_bytes = 0

    def initial_bifurcation(self, *args, **kwargs):
        counts = super().initial_bifurcation(*args, **kwargs)
        self.host_initial_write_bytes = sum(x.numel() * x.element_size()
                                            for x in (self.stt_k, self.stt_v))
        self.host_write_bytes += self.host_initial_write_bytes
        self.stt_k = self.stt_k.to('cpu')
        self.stt_v = self.stt_v.to('cpu')
        return counts

    def select_stt(self, tensor, idx):
        rows = self.index_stt_storage(tensor, idx)
        self.host_read_bytes += rows.numel() * rows.element_size()
        return rows.to(device=self.cfg.device, dtype=self.cfg.resolved_kv_dtype)

    def append_stt_storage(self, tensor, values):
        self.host_write_bytes += values.numel() * tensor.element_size()
        return super().append_stt_storage(
            tensor, values.to(device=tensor.device, dtype=tensor.dtype))


class CpuColdIndexSelectCache(CpuColdCache):
    """Experimental host gather using PyTorch's native index_select."""

    def index_stt_storage(self, tensor, idx):
        if tensor.device.type == 'cpu' and tensor.dtype.itemsize != 1:
            return torch.index_select(tensor, 1, torch.as_tensor(idx, dtype=torch.long))
        return super().index_stt_storage(tensor, idx)


def policy(mode):
    p = TieredKVPolicy(sram_budget=1024, stt_budget=2048,
                       start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
                       sttram_bifurcation_frac=.75,
                       inclusive=(mode != 'cpu_fast_no_shadow'))
    if mode == 'cpu':
        p.cache_cls = CpuColdCache
    elif mode in ('cpu_fast', 'cpu_fast_no_shadow'):
        p.cache_cls = CpuColdIndexSelectCache
    return p


def measure(model, tokenizer, prompt, tokens, ctx_len, decode, mode, warmup, repeat):
    p = policy(mode)
    out = profile_one(model, tokenizer, p, ctx_len, n_decode=decode,
                      n_warmup=warmup, n_repeat=repeat, device='cuda',
                      prompt_text=prompt, forced_token_ids=tokens)
    torch.cuda.synchronize()
    gc.collect()
    caches = list(p.layer_caches.values())
    out['final_gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
    out['cold_payload_bytes'] = sum(
        x.numel() * x.element_size()
        for c in caches for x in (c.stt_k, c.stt_v))
    out['cold_payload_device'] = str(caches[0].stt_k.device)
    if mode != 'gpu':
        out['explicit_h2d_payload_bytes'] = sum(c.host_read_bytes for c in caches)
        out['explicit_d2h_payload_bytes'] = sum(c.host_write_bytes for c in caches)
        out['initial_d2h_payload_bytes'] = sum(c.host_initial_write_bytes for c in caches)
        out['decode_d2h_payload_bytes'] = (
            out['explicit_d2h_payload_bytes'] - out['initial_d2h_payload_bytes'])
    p.reset()
    del p, caches
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='../models/mistral-7b-instruct')
    ap.add_argument('--prompt-file', default='results/remediation_v3/latency_prompt.txt')
    ap.add_argument('--stream-file', default='results/shadow_ablation_full/fixed_stream_20260929.json')
    ap.add_argument('--output', default='results/hardware/cpu_cold_probe_20260929.json')
    ap.add_argument('--ctx-len', type=int, default=4096)
    ap.add_argument('--decode', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--order', nargs='+',
                    choices=['gpu', 'cpu', 'cpu_fast', 'cpu_fast_no_shadow'],
                    default=['gpu', 'cpu'])
    args = ap.parse_args()
    prompt = Path(args.prompt_file).read_text()
    stream = json.loads(Path(args.stream_file).read_text())['config']['teacher_token_ids']
    assert len(stream) >= args.decode + 1
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = {}
    for i, name in enumerate(args.order):
        result = measure(model, tokenizer, prompt, stream, args.ctx_len,
                         args.decode, name, args.warmup, args.repeat)
        results[f'{i}_{name}'] = result
        print(name, 'ITL', result['itl_ms'], 'ms', 'cold device',
              result['cold_payload_device'], flush=True)
    same_policy = [r for name, r in results.items()
                   if not name.endswith('cpu_fast_no_shadow')]
    if len(same_policy) > 1:
        ref = same_policy[0]
        for candidate in same_policy[1:]:
            assert len(ref['trials']) == len(candidate['trials'])
            for trial_ref, trial_host in zip(ref['trials'], candidate['trials']):
                for key in ('predicted_token_ids', 'paid_writes', 'writes_saved',
                            'demoted', 'promoted', 'decode_steps'):
                    assert trial_ref[key] == trial_host[key], key
    cfg = dict(vars(args))
    cfg.update(runtime=runtime_identity(),
               prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
               stream_sha256=hashlib.sha256(json.dumps(stream).encode()).hexdigest(),
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               profile_sha256=hashlib.sha256(Path(__file__).with_name('latency_profile.py').read_bytes()).hexdigest(),
               tier='CPU DRAM pageable synchronous; not STT-RAM',
               byte_scope='explicit cold K/V GPU-CPU payload; excludes CPU compaction and allocator traffic')
    atomic_json(args.output, {'config': cfg, 'results': results,
                              'policy_parity_passed': len(same_policy) > 1})


if __name__ == '__main__':
    main()
