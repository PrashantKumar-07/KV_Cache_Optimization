#!/usr/bin/env python3
"""Measure how much true-query slow-tier relevance the current selector exposes."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import rotate_half
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity
from page_candidates import page_candidates, position_page_candidates, group_page_candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="../models/mistral-7b-instruct")
    ap.add_argument("--prompt-file", default="results/remediation_v3/latency_prompt.txt")
    ap.add_argument("--output", default="results/query_recall_probe.json")
    ap.add_argument("--ctx-len", type=int, default=16384)
    ap.add_argument("--decode", type=int, default=50)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--candidate-pages", type=int, default=0,
                    help="Positive enables offline page-candidate recall diagnostic")
    ap.add_argument("--page-bound", choices=("box", "l2", "centroid", "position_centroid",
                                             "stable_dense_centroid"),
                    default="box")
    args = ap.parse_args()
    prompt = Path(args.prompt_file).read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policy = TieredKVPolicy(sram_budget=1024, stt_budget=2048,
        start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
        sttram_bifurcation_frac=.75)
    records = []
    hooks = []
    page_assignments = {}

    def hook(layer):
        def observe(module, positional, kwargs):
            hidden = kwargs.get("hidden_states", positional[0] if positional else None)
            if hidden is None or hidden.size(1) != 1 or layer not in policy.layer_caches:
                return
            cache = policy.layer_caches[layer]
            live = torch.where(~cache.stt_shadow)[0]
            if not len(live):
                return
            cos, sin = kwargs["position_embeddings"]
            q = module.q_proj(hidden).view(1, 1, module.config.num_attention_heads,
                                            module.head_dim).transpose(1, 2)
            q = (q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1))
            groups = module.num_key_value_groups
            q = q[0, :, 0, :].float().view(cache.cfg.num_heads, groups, module.head_dim)
            k_live = cache.stt_k[:, live.to(cache.stt_k.device), :].float()
            k_fast = cache.sram_k.float()
            slow = torch.einsum("ghd,gnd->ghn", q, k_live)
            fast = torch.einsum("ghd,gnd->ghn", q, k_fast)
            weights = torch.softmax(torch.cat((fast, slow), dim=-1) * module.scaling,
                                    dim=-1)[..., fast.size(-1):].mean(dim=(0, 1))
            budget = min(policy.stt_expose_quota, len(live))
            oracle = torch.topk(weights, budget).indices
            positions = [cache.stt_pos[i] for i in live.tolist()]
            selected = set(policy._vis_stt_pos.get(layer, []))
            selected_idx = [i for i, pos in enumerate(positions) if pos in selected]
            oracle_idx = set(oracle.tolist())
            oracle_mass = weights[oracle].sum().item()
            selected_mass = weights[selected_idx].sum().item()
            records.append({"layer": layer, "step": policy.decode_step_count,
                "live": len(live), "selected": len(selected_idx),
                "oracle_overlap": len(oracle_idx & set(selected_idx)),
                "slow_mass": weights.sum().item(),
                "selected_mass": selected_mass, "oracle_mass": oracle_mass})
            if args.candidate_pages > 0:
                if args.page_bound == 'position_centroid':
                    candidates, bounds = position_page_candidates(q, k_live,
                        positions, args.page_size, args.candidate_pages)
                elif args.page_bound == 'stable_dense_centroid':
                    state = page_assignments.get(layer)
                    if state is None:
                        state = {'ids': {pos: i // args.page_size
                                         for i, pos in enumerate(cache.stt_pos)},
                                 'next_page': (len(cache.stt_pos) + args.page_size - 1)
                                              // args.page_size, 'next_fill': 0}
                        page_assignments[layer] = state
                    for pos in cache.stt_pos:
                        if pos not in state['ids']:
                            state['ids'][pos] = state['next_page']
                            state['next_fill'] += 1
                            if state['next_fill'] == args.page_size:
                                state['next_page'] += 1
                                state['next_fill'] = 0
                    candidates, bounds = group_page_candidates(q, k_live,
                        [state['ids'][pos] for pos in positions], args.candidate_pages)
                else:
                    candidates, bounds = page_candidates(q, k_live, args.page_size,
                                                         args.candidate_pages, args.page_bound)
                chosen = candidates[weights[candidates].topk(min(budget, len(candidates))).indices]
                candidate_logits = slow[..., candidates]
                local_weights = torch.softmax(torch.cat((fast, candidate_logits), dim=-1)
                    * module.scaling, dim=-1)[..., fast.size(-1):].mean(dim=(0, 1))
                local_chosen = candidates[local_weights.topk(min(budget, len(candidates))).indices]
                if args.page_bound in ('position_centroid', 'stable_dense_centroid'):
                    summary_bytes = (module.head_dim * cache.cfg.num_heads + 1) * 4 * bounds.shape[-1]
                    if args.page_bound == 'stable_dense_centroid':
                        summary_bytes += 4 * len(cache.stt_pos)  # stable page ID per retained row
                else:
                    n_values = (2 * module.head_dim if args.page_bound == 'box'
                        else module.head_dim + 1 if args.page_bound == 'l2'
                        else module.head_dim)
                    summary_bytes = n_values * 4 * cache.cfg.num_heads * bounds.shape[-1]
                records[-1].update(page_candidate_rows=len(candidates),
                    page_count=bounds.shape[-1],
                    page_oracle_overlap=len(oracle_idx & set(chosen.tolist())),
                    page_selected_mass=weights[chosen].sum().item(),
                    page_local_oracle_overlap=len(oracle_idx & set(local_chosen.tolist())),
                    page_local_selected_mass=weights[local_chosen].sum().item(),
                    page_underfilled=len(candidates) < budget,
                    page_summary_bytes=summary_bytes)
        return observe

    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        profile_one(model, tokenizer, policy, args.ctx_len, n_decode=args.decode,
                    n_warmup=0, n_repeat=1, device="cuda", prompt_text=prompt)
    finally:
        for handle in hooks:
            handle.remove()

    summary = {name: float(np.mean([r[name] for r in records]))
               for name in ("live", "selected", "oracle_overlap", "slow_mass",
                            "selected_mass", "oracle_mass")}
    summary["selected_over_oracle_mass"] = summary["selected_mass"] / summary["oracle_mass"]
    summary["selected_over_oracle_overlap"] = summary["oracle_overlap"] / summary["selected"]
    if args.candidate_pages > 0:
        for name in ('page_candidate_rows', 'page_oracle_overlap', 'page_selected_mass',
                     'page_local_oracle_overlap', 'page_local_selected_mass',
                     'page_summary_bytes', 'page_underfilled', 'page_count'):
            summary[name] = float(np.mean([r[name] for r in records]))
    atomic_json(args.output, {"config": {"model": args.model, "ctx_len": args.ctx_len,
        "decode": args.decode, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "runtime": runtime_identity(),
        "page_size": args.page_size, "candidate_pages": args.candidate_pages,
        "page_bound": args.page_bound,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "page_selector_sha256": hashlib.sha256(Path(__file__).with_name('page_candidates.py').read_bytes()).hexdigest(),
        "page_scope": ("offline max-head page bound" if args.page_bound in ('box', 'l2')
                       else "offline max-head centroid ranking") +
                      "; exact reranking uses oracle full-key softmax denominators; no deployable quality/latency claim",
        "scope": "actual decode Q versus all live STT K; softmax includes VRAM and STT but omits new self token"},
        "summary": summary, "records": records})
    print(summary)


if __name__ == "__main__":
    main()
