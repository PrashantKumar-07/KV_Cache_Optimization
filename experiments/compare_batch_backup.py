"""ABBA screen for batched shadow-backup metadata restoration."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--prompt-file',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--ctx',type=int,default=16384)
    ap.add_argument('--decode',type=int,default=16)
    ap.add_argument('--blocks',type=int,default=3)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map=args.device,
        attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(args.model)
    prompt=Path(args.prompt_file).read_text()

    def policy(fast):
        return TieredKVPolicy(sram_budget=1024,stt_budget=2048,
                              start_size=4,recent_size=128,
                              sttram_bifurcation_frac=.75,
                              kv_dtype=torch.bfloat16,
                              batch_backup_updates=fast)

    profile_one(model,tok,policy(False),args.ctx,args.decode,0,1,args.device,prompt)
    rows=[]
    for block in range(args.blocks):
        for arm in ('control','fast','fast','control'):
            gc.collect()
            stats=profile_one(model,tok,policy(arm=='fast'),args.ctx,
                              args.decode,0,1,args.device,prompt)
            row={'block':block,'arm':arm,'itl_ms':stats['itl_ms'],
                 'forward_ms':stats['forward_ms'],
                 'peak_gb':stats['peak_memory_gb'],
                 'step_ms':stats['trials'][0]['step_ms']}
            rows.append(row)
            print({k:v for k,v in row.items() if k!='step_ms'},flush=True)
    summary={}
    for arm in ('control','fast'):
        selected=[r for r in rows if r['arm']==arm]
        steps=[v for r in selected for v in r['step_ms']]
        summary[arm]={'median_itl_ms':statistics.median(steps),
                      'p95_itl_ms':float(np.quantile(steps,.95)),
                      'median_forward_ms':statistics.median(r['forward_ms'] for r in selected),
                      'max_peak_gb':max(r['peak_gb'] for r in selected)}
    pairs=[]
    for block in range(args.blocks):
        base=statistics.median(r['itl_ms'] for r in rows
                               if r['block']==block and r['arm']=='control')
        fast=statistics.median(r['itl_ms'] for r in rows
                               if r['block']==block and r['arm']=='fast')
        pairs.append({'block':block,'control_ms':base,'fast_ms':fast,
                      'improvement_pct':100*(base-fast)/base})
    result={'scope':'same-process real-text ABBA full model-policy decode',
            'config':vars(args),'runtime':runtime_identity(),
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'rows':rows,'summary':summary,'paired_blocks':pairs,
            'limitations':'One model/prompt/machine; paired capped-generation parity is separate.'}
    atomic_json(args.output,result)
    print(json.dumps({'summary':summary,'paired_blocks':pairs},indent=2))


if __name__=='__main__': main()
