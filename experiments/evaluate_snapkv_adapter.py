"""LongBench pilot using upstream SnapKV selection and expanded-head storage."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import longbench_eval as bench
from snapkv_adapter import SnapKVAdapter
from provenance import atomic_json, runtime_identity, resume_results, model_artifact_identity


@torch.inference_mode()
def generate(model, tokenizer, input_text, max_gen, policy, device, max_ctx=31500, task_name='', *, prefill_last_logits=False):
    ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    if ids.shape[1] > max_ctx:
        half = max_ctx // 2
        ids = torch.cat((ids[:, :half], ids[:, -(max_ctx-half):]), 1)
    length = ids.shape[1]
    policy.last_prompt_identity = {
        'prompt_sha256': hashlib.sha256(input_text.encode()).hexdigest(),
        'input_ids_sha256': hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
        'max_ctx': max_ctx}
    cache = SnapKVAdapter(model, policy.capacity, policy.window)
    generated = []
    try:
        out = model(ids, past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(length, device=device).unsqueeze(0),
                    **bench.prefill_logit_kwargs(model, prefill_last_logits))
        for step in range(max_gen):
            token = out.logits[:, -1:].argmax(-1)
            generated.append(token.item())
            if token.item() == tokenizer.eos_token_id or step + 1 == max_gen:
                break
            del out  # unused prefill logits must not survive the decode allocation
            out = model(token, past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[length+step]], device=device))
        payload = sum(x.numel()*x.element_size() for layer in cache.layers
                      for x in (layer.keys, layer.values))
        policy.last_generation = {'prompt_tokens': length, 'generated_tokens': len(generated),
                                  'final_kv_payload_bytes': payload}
    finally:
        cache.close()
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='../models/mistral-7b-instruct')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--max-samples', type=int, default=5)
    p.add_argument('--capacity', type=int, default=1024)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--attn-implementation', choices=('sdpa', 'eager'), default='sdpa')
    p.add_argument('--output', required=True)
    p.add_argument('--prefill-last-logits', action='store_true')
    args = p.parse_args()
    config = vars(args).copy()
    config.pop('output')
    config['runtime'] = runtime_identity()
    config['model_identity'] = model_artifact_identity(args.model)
    config['seed'] = 0
    config['max_ctx'] = 31500
    config['source_files'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name('snapkv_adapter.py'),
                     Path('sota/snapkv/snapkv/monkeypatch/snapkv_utils.py'))}
    config['dataset_sha256'] = {t: hashlib.sha256(bench.task_data_path(args.data_dir,t).read_bytes()).hexdigest()
                                for t in args.tasks}
    results = resume_results(args.output, config)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                device_map='cuda', attn_implementation=args.attn_implementation).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original = bench.generate_with_policy
    bench.generate_with_policy = generate
    try:
        for task in args.tasks:
            if task in results:
                continue
            policy = SimpleNamespace(name='SnapKV upstream selector', capacity=args.capacity, window=args.window)
            score, samples = bench.evaluate_task(model, tokenizer, task, policy, 'cuda',
                args.max_samples, True, max_ctx=31500, data_dir=args.data_dir,
                prefill_last_logits=args.prefill_last_logits)
            results[task] = {'snapkv_upstream_selector': {'score': score, 'n':len(samples),
                'sample_scores': samples, 'samples':policy.sample_records}}
            atomic_json(args.output, {'config':config, 'results':results})
    finally:
        bench.generate_with_policy = original


if __name__ == '__main__':
    main()
