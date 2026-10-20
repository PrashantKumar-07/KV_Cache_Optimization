#!/usr/bin/env python3
"""Paired research probe: select visible cold KV with the current layer query."""
import argparse
import hashlib
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import rotate_half
import longbench_eval as bench
from page_candidates import page_candidates
from stable_page_index import IndexedTieredKVCache
from provenance import atomic_json, resume_results, runtime_identity


class MeasuredTieredPolicy(bench.TieredKVPolicy):
    def __call__(self, past_kv):
        first = not self.initialized
        result = super().__call__(past_kv)
        if first and self.request_start_s is not None:
            torch.cuda.synchronize()
            self.prefill_s = time.perf_counter() - self.request_start_s
            self.prefill_peak_alloc_bytes = torch.cuda.max_memory_allocated()
        return result


class QuerySelector:
    """Replace equal-length visible STT rows just before each attention layer."""
    def __init__(self, model, policy, candidate_pages=0, use_index=False):
        self.policy = policy
        self.candidate_pages = candidate_pages
        self.use_index = use_index
        self.handles = [layer.self_attn.register_forward_pre_hook(
            self._hook(index), with_kwargs=True)
            for index, layer in enumerate(model.model.layers)]

    def close(self):
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer_index):
        def select(module, positional, kwargs):
            hidden = kwargs.get("hidden_states", positional[0] if positional else None)
            if hidden is None or hidden.size(1) != 1 or layer_index not in self.policy.layer_caches:
                return
            cache = self.policy.layer_caches[layer_index]
            old_positions = self.policy._vis_stt_pos.get(layer_index, [])
            if not old_positions:
                return
            live = torch.where(~cache.stt_shadow)[0].to(cache.stt_k.device)
            if len(live) <= len(old_positions):
                return
            cos, sin = kwargs["position_embeddings"]
            q = module.q_proj(hidden).view(1, 1, module.config.num_attention_heads,
                                            module.head_dim).transpose(1, 2)
            q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            q = q[0, :, 0, :].float().view(cache.cfg.num_heads,
                    module.num_key_value_groups, module.head_dim)
            fast_idx = self.policy._vis_sram_keep[layer_index]
            fast_k = cache.sram_k[:, fast_idx, :].float()
            if self.use_index:
                rows = cache.page_index.candidate_rows(q, cache, live,
                    self.candidate_pages, len(old_positions))
                candidates = live.index_select(0, rows)
                live_k = cache.stt_k.index_select(1, candidates).float()
            else:
                live_k = cache.stt_k.index_select(1, live).float()
            if self.candidate_pages and not self.use_index:
                rows, _ = page_candidates(q, live_k, 16, self.candidate_pages,
                                          bound_type='centroid')
                candidates = live.index_select(0, rows)
                live_k = live_k.index_select(1, rows)
            elif not self.use_index:
                candidates = live
            fast_logits = torch.einsum("ghd,gnd->ghn", q, fast_k)
            slow_logits = torch.einsum("ghd,gnd->ghn", q, live_k)
            weights = torch.softmax(torch.cat((fast_logits, slow_logits), dim=-1)
                                    * module.scaling, dim=-1)[..., fast_k.size(1):]
            ranking = weights.mean(dim=(0, 1))
            chosen = candidates[torch.topk(ranking, len(old_positions)).indices].tolist()
            chosen.sort()
            layer = kwargs["past_key_values"].layers[layer_index]
            keys = torch.cat((cache.sram_k[:, fast_idx, :], cache.stt_k[:, chosen, :]), dim=1)
            values = torch.cat((cache.sram_v[:, fast_idx, :], cache.stt_v[:, chosen, :]), dim=1)
            if keys.size(1) != layer.keys.size(-2):
                raise RuntimeError("Query selection changed shared causal-mask length")
            layer.keys = keys.unsqueeze(0).to(layer.keys.dtype)
            layer.values = values.unsqueeze(0).to(layer.values.dtype)
            self.policy._vis_stt_pos[layer_index] = [cache.stt_pos[i] for i in chosen]
        return select


class PromptQueryCapture:
    """Capture only rotary-positioned observation queries before compression."""
    def __init__(self, model, policy):
        self.handles = []
        for index, layer in enumerate(model.model.layers):
            def capture(module, positional, kwargs, index=index):
                if policy.initialized:
                    return
                hidden = kwargs.get("hidden_states", positional[0] if positional else None)
                if hidden is None or hidden.size(0) != 1:
                    raise ValueError("Prompt query capture requires batch size one")
                hidden = hidden[:, -policy.recent_size:]
                shape = (1, hidden.size(1), module.config.num_attention_heads, module.head_dim)
                q = module.q_proj(hidden).view(shape).transpose(1, 2)
                cos, sin = kwargs["position_embeddings"]
                cos, sin = cos[:, -hidden.size(1):], sin[:, -hidden.size(1):]
                policy.prompt_q_by_layer[index] = (q * cos.unsqueeze(1) +
                    rotate_half(q) * sin.unsqueeze(1))[0]
            self.handles.append(layer.self_attn.register_forward_pre_hook(
                capture, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="../models/mistral-7b-instruct")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--max-samples", type=int, default=5)
    p.add_argument("--strategy", choices=("lru", "query", "centroid", "indexed_centroid"),
                   required=True)
    p.add_argument("--candidate-pages", type=int, default=32,
                   help="Sixteen-row pages ranked by centroid in centroid mode")
    p.add_argument("--tiered-vram-budget", type=int, default=1024)
    p.add_argument("--tiered-stt-budget", type=int, default=2048)
    p.add_argument("--stt-expose-quota", type=int, default=32)
    p.add_argument("--real-query-prefill", action="store_true",
                   help="Opt-in causal importance from actual rotary prompt queries")
    p.add_argument("--query-score-mode", choices=("chunked", "expanded"),
                   default="chunked", help="Actual-query score implementation")
    p.add_argument("--query-chunk-size", type=int, default=128,
                   help="Observation queries per GQA score pass")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.strategy in ('centroid', 'indexed_centroid') and args.candidate_pages < 1:
        p.error('--candidate-pages must be positive in centroid mode')
    config = {"model": args.model, "data_dir": args.data_dir,
        "tasks": args.tasks, "max_samples": args.max_samples,
        "strategy": args.strategy, "seed": 0, "max_ctx": 31500,
        "candidate_pages": args.candidate_pages if args.strategy in
                           ('centroid', 'indexed_centroid') else 0,
        "tiered_vram_budget": args.tiered_vram_budget,
        "tiered_stt_budget": args.tiered_stt_budget,
        "recent": 128, "stt_expose_quota": args.stt_expose_quota,
        "real_query_prefill": args.real_query_prefill,
        "query_score_mode": args.query_score_mode,
        "query_chunk_size": args.query_chunk_size,
        "sttram_bifurcation_frac": .75, "kv_dtype": "bfloat16",
        "runtime": runtime_identity(),
        "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "page_index_sha256": (hashlib.sha256(Path(__file__).with_name(
            'stable_page_index.py').read_bytes()).hexdigest()
            if args.strategy == 'indexed_centroid' else None),
        "page_candidates_sha256": (hashlib.sha256(Path(__file__).with_name(
            'page_candidates.py').read_bytes()).hexdigest()
            if args.strategy in ('centroid', 'indexed_centroid') else None),
        "dataset_sha256": {t: hashlib.sha256(bench.task_data_path(args.data_dir,t).read_bytes()).hexdigest()
                           for t in args.tasks}}
    results = resume_results(args.output, config)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model,
        torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for task in args.tasks:
        if task in results:
            continue
        policy = MeasuredTieredPolicy(sram_budget=args.tiered_vram_budget,
            stt_budget=args.tiered_stt_budget, stt_expose_quota=args.stt_expose_quota,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
            sttram_bifurcation_frac=.75)
        policy.prompt_query_score_mode = args.query_score_mode
        policy.prompt_query_chunk = args.query_chunk_size
        if args.strategy == 'indexed_centroid':
            policy.cache_cls = IndexedTieredKVCache
        selector = (QuerySelector(model, policy,
                    candidate_pages=args.candidate_pages if args.strategy in
                        ('centroid', 'indexed_centroid') else 0,
                    use_index=args.strategy == 'indexed_centroid')
                    if args.strategy in ('query', 'centroid', 'indexed_centroid') else None)
        capture = PromptQueryCapture(model, policy) if args.real_query_prefill else None
        original_generate = bench.generate_with_policy

        def generate_with_bytes(*call_args, **call_kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            policy.request_start_s = time.perf_counter()
            prediction = original_generate(*call_args, **call_kwargs)
            policy.last_generation["final_resident_kv_payload_bytes"] = sum(
                tensor.numel() * tensor.element_size()
                for cache in policy.layer_caches.values()
                for tensor in (cache.sram_k, cache.sram_v, cache.stt_k, cache.stt_v))
            policy.last_generation["prefill_s"] = policy.prefill_s
            policy.last_generation["prefill_peak_alloc_bytes"] = policy.prefill_peak_alloc_bytes
            if args.strategy == 'indexed_centroid':
                for cache in policy.layer_caches.values():
                    cache.page_index.validate(cache)
                policy.last_generation['page_index_logical_bytes'] = sum(
                    cache.page_index.sums.numel() * cache.page_index.sums.element_size() +
                    cache.page_index.counts.numel() * cache.page_index.counts.element_size() +
                    4 * len(cache.stt_pos) for cache in policy.layer_caches.values())
                policy.last_generation['page_index_changed_key_bytes'] = sum(
                    cache.page_index.changed_key_bytes for cache in policy.layer_caches.values())
            return prediction

        bench.generate_with_policy = generate_with_bytes
        try:
            score, sample_scores = bench.evaluate_task(model, tokenizer, task,
                policy, "cuda", args.max_samples, True, max_ctx=31500,
                data_dir=args.data_dir)
        finally:
            bench.generate_with_policy = original_generate
            if selector:
                selector.close()
            if capture:
                capture.close()
        results[task] = {"tieredkv": {"score": score, "n": len(sample_scores),
            "sample_scores": sample_scores, "samples": policy.sample_records,
            "cache_stats": policy.stats()}}
        atomic_json(args.output, {"config": config, "results": results})
        print(task, args.strategy, score, flush=True)


if __name__ == "__main__":
    main()
