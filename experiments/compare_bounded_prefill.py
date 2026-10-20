#!/usr/bin/env python3
"""Measure peak HBM and TTFT for exact versus chunk-compressed prefill."""
import argparse
import hashlib
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from longbench_eval import TieredKVPolicy, generate_with_policy
from provenance import atomic_json, runtime_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='../models/mistral-7b-instruct')
    p.add_argument('--prompt-file',default='results/remediation_v3/latency_prompt.txt')
    p.add_argument('--output',default='results/bounded_prefill/pilot.json')
    p.add_argument('--ctx-len',type=int,default=16384)
    p.add_argument('--chunk-size',type=int,default=512)
    args=p.parse_args()
    prompt=Path(args.prompt_file).read_text()
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        device_map='cuda',attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(args.model)
    # Freeze the exact tokenizer prefix used by both arms.
    ids=tok(prompt,return_tensors='pt').input_ids[:,:args.ctx_len]
    prompt=tok.decode(ids[0],skip_special_tokens=False)
    results={}
    for i,name in enumerate(('exact','chunked','chunked','exact')):
        policy=TieredKVPolicy(sram_budget=1024,stt_budget=2048,start_size=4,
            recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75,
            prefill_chunk_size=(args.chunk_size if name=='chunked' else 0))
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started=time.perf_counter()
        prediction=generate_with_policy(model,tok,prompt,1,policy,'cuda',max_ctx=args.ctx_len)
        torch.cuda.synchronize()
        elapsed=(time.perf_counter()-started)*1000
        resident=sum(t.numel()*t.element_size() for c in policy.layer_caches.values()
                     for t in (c.sram_k,c.sram_v,c.stt_k,c.stt_v))
        layer_counts={str(i):{'sram':c.sram_k.shape[1],'stt':c.stt_k.shape[1]}
                      for i,c in policy.layer_caches.items()}
        results[f'{i}_{name}']={'ttft_ms':elapsed,
            'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,
            'allocated_after_gb':torch.cuda.memory_allocated()/1e9,
            'resident_kv_bytes':resident,'prediction':prediction,
            'prompt_tokens':policy.last_generation['prompt_tokens'],
            'layer_counts':layer_counts}
        print(i,name,elapsed,results[f'{i}_{name}']['peak_allocated_gb'],flush=True)
        del policy
    atomic_json(args.output,{'config':{'model':args.model,'ctx_len':args.ctx_len,
        'chunk_size':args.chunk_size,'order':['exact','chunked','chunked','exact'],
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'runtime':runtime_identity()},'results':results,
        'scope':'Approximate chunked prefill: later chunks attend only compressed prior KV.'})


if __name__=='__main__': main()
