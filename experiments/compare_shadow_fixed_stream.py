"""ABBA shadow ablation with identical teacher-forced decode tokens."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latency_profile import profile_one
from longbench_eval import TieredKVPolicy, _attn_impl
from provenance import atomic_json, runtime_identity


def teacher_stream(model, ids, count):
    """Greedy full-cache tokens, including the first token after prefill."""
    with torch.inference_mode(), _attn_impl(model, "sdpa"):
        out = model(ids, use_cache=True,
                    position_ids=torch.arange(ids.size(1), device=ids.device)[None])
    past = out.past_key_values
    token = out.logits[:, -1:].argmax(dim=-1)
    tokens = [token.item()]
    for step in range(count):
        with torch.inference_mode():
            out = model(token, past_key_values=past, use_cache=True,
                        position_ids=torch.tensor([[ids.size(1) + step]], device=ids.device))
        past = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        tokens.append(token.item())
    del out, past, token
    gc.collect()
    torch.cuda.empty_cache()
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="../models/mistral-7b-instruct")
    parser.add_argument("--prompt-file", default="results/remediation_v3/latency_prompt.txt")
    parser.add_argument("--output", default="results/shadow_ablation_full/fixed_stream_20260929.json")
    parser.add_argument("--ctx-len", type=int, default=16384)
    parser.add_argument("--decode", type=int, default=50)
    args = parser.parse_args()
    prompt = Path(args.prompt_file).read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[:, :args.ctx_len].to("cuda")
    assert ids.size(1) == args.ctx_len
    tokens = teacher_stream(model, ids, args.decode)
    assert len(tokens) == args.decode + 1
    token_sha = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    del ids
    results = {}
    for index, name in enumerate(("inclusive", "no_shadow", "no_shadow", "inclusive")):
        policy = TieredKVPolicy(sram_budget=1024, stt_budget=2048,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
            sttram_bifurcation_frac=.75, inclusive=name == "inclusive")
        stats = profile_one(model, tokenizer, policy, args.ctx_len,
            n_decode=args.decode, n_warmup=1, n_repeat=2,
            device="cuda", prompt_text=prompt, forced_token_ids=tokens)
        assert len(stats["trials"]) == 2
        assert all(trial["decode_steps"] == args.decode for trial in stats["trials"])
        results[f"{index}_{name}"] = stats
        print(index, name, [r["paid_writes"] for r in stats["trials"]], flush=True)
    atomic_json(args.output, {"config": {"model": args.model, "ctx_len": args.ctx_len,
        "decode": args.decode, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "teacher_token_ids": tokens, "teacher_token_sha256": token_sha,
        "order": ["inclusive", "no_shadow", "no_shadow", "inclusive"],
        "runtime": runtime_identity(),
        "profile_script_sha256": hashlib.sha256(Path(__file__).with_name("latency_profile.py").read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_scope": "end_to_end_model_policy_teacher_forced_fixed_stream"},
        "results": results})


if __name__ == "__main__":
    main()
