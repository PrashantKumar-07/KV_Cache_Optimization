#!/usr/bin/env python3
"""Teacher-forced, token-by-token cache-policy perplexity.

Each input token predicts the next token exactly once. Policies receive real
attention feedback before eviction. Each context checkpoint aggregates NLL
and token counts over all documents reaching it; short documents are excluded
from that checkpoint, never relabeled as longer contexts. This deliberately
uses single-token decode to match the generation policy semantics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from longbench_eval import to_tuple_kv, apply_tuple_kv_inplace, _attn_impl
from latency_profile import make_policy
from provenance import atomic_json, read_jsonl, resume_results, runtime_identity


@torch.inference_mode()
def evaluate_ppl(model, tokenizer, texts, policy, budget=None, device='cpu',
                 max_len=32000, stride=1, name='method', checkpoints=None):
    if stride != 1:
        raise ValueError('PPL requires stride=1 so all policies receive single-token feedback')
    points = sorted(set(checkpoints or [1024,2048,4096,8192,16384,32000]))
    points = [c for c in points if 0 < c <= max_len]
    totals = {c: {'nll':0.0, 'tokens':0, 'documents':0} for c in points}
    for text in texts:
        ids = tokenizer(text, return_tensors='pt', truncation=False).input_ids.to(device)
        length = min(ids.size(1)-1, max_len)
        if length < 1:
            continue
        policy.reset()
        past = None
        nll = 0.0
        with _attn_impl(model, 'eager'):
            for i in range(length):
                out = model(ids[:, i:i+1], past_key_values=past, use_cache=True,
                    position_ids=torch.tensor([[i]], device=device), output_attentions=True)
                loss = torch.nn.functional.cross_entropy(
                    out.logits[:, -1, :].float(), ids[:, i+1], reduction='sum')
                nll += float(loss)
                past = out.past_key_values
                tup = to_tuple_kv(past)
                if hasattr(policy, 'update_scores'):
                    policy.update_scores(tup, out.attentions)
                apply_tuple_kv_inplace(past, policy(tup))
                count = i+1
                if count in totals:
                    totals[count]['nll'] += nll
                    totals[count]['tokens'] += count
                    totals[count]['documents'] += 1
        del past, out
    return {str(c): {**v, 'ppl':math.exp(v['nll']/v['tokens'])}
            for c,v in totals.items() if v['tokens']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--texts-jsonl', required=True, help='JSONL with context or text fields; no synthetic fallback')
    p.add_argument('--output', default='results/remediation/ppl.json')
    p.add_argument('--max-len', type=int, default=32000)
    p.add_argument('--checkpoints', nargs='+', type=int, default=[1024,2048,4096,8192,16384,32000])
    p.add_argument('--n-docs', type=int, default=20)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--methods', nargs='+', default=['full','streamingllm','h2o','snapkv','tieredkv'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    args=p.parse_args()
    raw=Path(args.texts_jsonl).read_bytes()
    texts=[]
    for row in read_jsonl(args.texts_jsonl):
        text=row.get('context',row.get('text'))
        if not isinstance(text,str): raise ValueError('Missing context/text string')
        texts.append(text)
        if len(texts)>=args.n_docs: break
    if args.n_docs<1 or not texts: raise ValueError('At least one document required')
    config={k:v for k,v in vars(args).items() if k!='output'}
    config.update(runtime=runtime_identity(), dataset_sha256=hashlib.sha256(raw).hexdigest(),
                  protocol='teacher_forced_single_token_v2', tiered_budgets=[args.budget,2*args.budget],
                  tiered_dtype='bfloat16', sttram_bifurcation_frac=.75)
    results=resume_results(args.output,config)
    torch.manual_seed(args.seed)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        device_map=args.device,attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(args.model)
    for method in args.methods:
        if method in results: continue
        policy=make_policy(method,args.budget)
        results[method]=evaluate_ppl(model,tok,texts,policy,device=args.device,
            max_len=args.max_len,checkpoints=args.checkpoints)
        atomic_json(args.output,{'config':config,'results':results})
        print(method,results[method],flush=True)


if __name__=='__main__':
    main()
