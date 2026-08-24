#!/usr/bin/env python3
"""Quick debug: what is the model actually generating on LongBench samples?"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from longbench_metrics import DATASET_TO_PROMPT, DATASET_TO_MAXGEN, DATASET_TO_METRIC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

print(f"Loading model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

task = "multifieldqa_en"
data = load_dataset("THUDM/LongBench", task, split="test")
prompt_fmt = DATASET_TO_PROMPT[task]
max_gen = DATASET_TO_MAXGEN[task]
metric_fn = DATASET_TO_METRIC[task]

# Test first 3 samples with NO eviction (full cache)
for i, sample in enumerate(list(data)[:3]):
    prompt = prompt_fmt.format(**sample)
    print(f"\n{'='*60}")
    print(f"Sample {i}")
    print(f"Question: {sample['input'][:200]}")
    print(f"Ground truth answers: {sample['answers']}")
    print(f"Prompt length (chars): {len(prompt)}")

    # Tokenize
    toks = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = toks.input_ids
    print(f"Token count: {input_ids.size(1)}")

    # Truncate from middle if needed
    max_ctx = 4096
    if input_ids.size(1) > max_ctx:
        half = max_ctx // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
        print(f"Truncated to: {input_ids.size(1)} tokens")

    input_ids = input_ids.to(args.device)

    # Generate with standard HF generate (no custom policy)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, max_new_tokens=max_gen, do_sample=False)

    # Decode only the NEW tokens
    new_tokens = output_ids[0, input_ids.size(1):]
    pred_hf = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"HF generate output: [{pred_hf}]")

    # Now test OUR custom generate loop (same as longbench_eval.py)
    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(dim=-1)
        generated = [next_tok.item()]

        for _ in range(max_gen - 1):
            out = model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            tok_id = next_tok.item()
            generated.append(tok_id)
            if tok_id == tokenizer.eos_token_id:
                break

    pred_custom = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Custom loop output: [{pred_custom}]")

    # Score both
    for gt in sample["answers"]:
        score_hf = metric_fn(pred_hf, gt)
        score_custom = metric_fn(pred_custom, gt)
        print(f"  vs GT [{gt[:100]}...]: HF_score={score_hf:.4f}  Custom_score={score_custom:.4f}")
