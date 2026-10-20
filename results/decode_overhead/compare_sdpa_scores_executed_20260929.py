"""Opt-in SDPA decode with explicitly reconstructed policy attention scores."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import repeat_kv, rotate_half

from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity


class ScoreProvider:
    """Capture the rotary query, then score the cache returned by that forward."""

    def __init__(self, model):
        self.queries = {}
        self.layers = model.model.layers
        self.handles = [layer.self_attn.register_forward_pre_hook(
            self._hook(i), with_kwargs=True) for i, layer in enumerate(self.layers)]

    def _hook(self, index):
        def capture(module, positional, kwargs):
            hidden = kwargs.get('hidden_states', positional[0] if positional else None)
            if hidden is None or hidden.size(1) != 1:
                return
            cos, sin = kwargs['position_embeddings']
            q = module.q_proj(hidden).view(
                1, 1, module.config.num_attention_heads, module.head_dim).transpose(1, 2)
            self.queries[index] = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
        return capture

    def weights(self, past_kv):
        if len(self.queries) != len(past_kv):
            raise RuntimeError('Missing captured query for SDPA score reconstruction')
        result = []
        for i, (k, _) in enumerate(past_kv):
            q = self.queries[i]
            keys = repeat_kv(k, q.size(1) // k.size(1))
            logits = torch.matmul(q, keys.transpose(2, 3)) * self.layers[i].self_attn.scaling
            result.append(torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype))
        return tuple(result)

    def close(self):
        for handle in self.handles:
            handle.remove()


def state_digest(policy):
    state = []
    for layer in sorted(policy.layer_caches):
        cache = policy.layer_caches[layer]
        state.append((cache.sram_pos, cache.stt_pos, cache.stt_shadow.tolist(),
                      cache.stt_was_vram_resident.tolist()))
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='../models/mistral-7b-instruct')
    ap.add_argument('--prompt-file', default='results/remediation_v3/latency_prompt.txt')
    ap.add_argument('--stream-file', default='results/shadow_ablation_full/fixed_stream_20260929.json')
    ap.add_argument('--output', default='results/decode_overhead/sdpa_scores_abba_20260929.json')
    ap.add_argument('--ctx-len', type=int, default=4096)
    ap.add_argument('--decode', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--order', nargs='+', choices=['eager', 'sdpa'],
                    default=['eager', 'sdpa'])
    args = ap.parse_args()
    prompt = Path(args.prompt_file).read_text()
    tokens = json.loads(Path(args.stream_file).read_text())['config']['teacher_token_ids']
    assert len(tokens) >= args.decode + 1
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = {}
    for index, name in enumerate(args.order):
        p = TieredKVPolicy(sram_budget=1024, stt_budget=2048,
                           start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
                           sttram_bifurcation_frac=.75)
        provider = ScoreProvider(model) if name == 'sdpa' else None
        try:
            out = profile_one(model, tokenizer, p, args.ctx_len,
                              n_decode=args.decode, n_warmup=args.warmup,
                              n_repeat=args.repeat, device='cuda',
                              prompt_text=prompt, forced_token_ids=tokens,
                              score_provider=provider)
        finally:
            if provider is not None:
                provider.close()
        out['final_state_sha256'] = state_digest(p)
        results[f'{index}_{name}'] = out
        print(name, out['itl_ms'], flush=True)
        p.reset()
        del p, out
        gc.collect()
        torch.cuda.empty_cache()
    reference = next(iter(results.values()))
    parity = all(candidate['final_state_sha256'] == reference['final_state_sha256'] and
        len(candidate['trials']) == len(reference['trials']) and
        all(all(a[key] == b[key] for key in ('predicted_token_ids', 'paid_writes',
            'writes_saved', 'demoted', 'promoted', 'decode_steps'))
            for a, b in zip(reference['trials'], candidate['trials']))
        for candidate in results.values())
    config = vars(args)
    config.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  teacher_token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  profile_sha256=hashlib.sha256(Path(__file__).with_name('latency_profile.py').read_bytes()).hexdigest(),
                  runtime=runtime_identity(),
                  scope='same-process full-loop fixed-token model+policy timing')
    atomic_json(args.output, {'config': config, 'results': results,
                              'policy_output_parity': parity})
    if not parity:
        raise AssertionError('SDPA candidate changed predictions or cache policy state')


if __name__ == '__main__':
    main()
