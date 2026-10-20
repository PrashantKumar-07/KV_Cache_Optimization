"""Paired real-prompt generation parity for batched shadow restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longbench_eval import (DATASET_TO_PROMPT, TieredKVPolicy,
                            generate_with_policy, task_data_path)
from provenance import atomic_json, read_jsonl, runtime_identity


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='../models/mistral-7b-instruct')
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--samples',type=int,default=5)
    ap.add_argument('--max-gen',type=int,default=32)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map=args.device,
        attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(args.model)
    tasks=('qasper','narrativeqa','gov_report')
    hashes={task:hashlib.sha256(task_data_path(args.data_dir,task).read_bytes()).hexdigest()
            for task in tasks}
    rows=[]
    for task in tasks:
        items=read_jsonl(task_data_path(args.data_dir,task))[:args.samples]
        for sample,item in enumerate(items):
            prompt='[INST] '+DATASET_TO_PROMPT[task].format(**item)+' [/INST]'
            predictions={};counters={}
            for arm,batched in (('row_updates',False),('batched',True)):
                policy=TieredKVPolicy(sram_budget=1024,stt_budget=2048,
                                      start_size=4,recent_size=128,
                                      sttram_bifurcation_frac=.75,
                                      kv_dtype=torch.bfloat16,
                                      batch_backup_updates=batched)
                with torch.inference_mode():
                    predictions[arm]=generate_with_policy(
                        model,tok,prompt,args.max_gen,policy,args.device,
                        max_ctx=31500,task_name=task)
                counters[arm]=policy.stats()
            row={'task':task,'sample':sample,
                 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                 'equal_prediction':predictions['row_updates']==predictions['batched'],
                 'predictions':predictions,
                 'saved_writes':{arm:counters[arm]['writes_saved'] for arm in counters},
                 'paid_writes':{arm:counters[arm]['paid_writes'] for arm in counters}}
            rows.append(row)
            print(task,sample,row['equal_prediction'],row['saved_writes'],flush=True)
            atomic_json(args.output,{'scope':'paired first-N LongBench prompts; generation capped for token parity, not task scoring',
                                     'config':vars(args),'runtime':runtime_identity(),
                                     'dataset_sha256':hashes,'rows':rows,
                                     'all_equal':all(r['equal_prediction'] for r in rows),
                                     'limitations':'Capped 32-token generation; exact prediction parity only, not a full quality score.'})
    if not all(r['equal_prediction'] for r in rows):
        raise AssertionError('generation parity failure')


if __name__=='__main__': main()
