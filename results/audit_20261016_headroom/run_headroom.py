"""Fixed headroom mechanism ablation, reusing the unchanged full-run generator.

All rows are descriptive: the previous complete benchmark has already been
observed. No new generalization, operating-point tuning or SOTA claim is made.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FULL = ROOT / 'results/full_evaluation_20261015'
spec = importlib.util.spec_from_file_location('frozen_full_generator', FULL / 'run_full.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
torch = frozen.torch
bench = frozen.bench
from provenance import atomic_json, read_jsonl, model_artifact_identity, runtime_identity


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def fraction_policy(fraction):
    """Change only the constructor's explicit fraction; restore module binding."""
    original = frozen.policy_for
    def constructor(arm):
        assert arm == 'tiered_hot1024'
        return bench.TieredKVPolicy(sram_budget=1024, stt_budget=2048,
            stt_expose_quota=32, start_size=4, recent_size=128,
            kv_dtype=torch.bfloat16, sttram_bifurcation_frac=fraction)
    frozen.policy_for = constructor
    try:
        yield
    finally:
        frozen.policy_for = original


def run_request(model, tokenizer, ids, task, item, fraction):
    with fraction_policy(fraction):
        return frozen.run_request(model, tokenizer, ids, task, item, 'tiered_hot1024')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    assert args.shard in (0, 1)
    plan_path = BASE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    assert sha(__file__) == plan['runner_sha256']
    assert sha(FULL / 'run_full.py') == plan['frozen_generator_sha256']
    assert sha(FULL / 'plan.json') == plan['reference_plan_sha256']
    assert sha(FULL / 'verified_results.json') == plan['reference_verification_sha256']
    assert all(sha(ROOT / p) == digest for p, digest in plan['source_sha256'].items())
    assert model_artifact_identity(plan['model']) == plan['model_identity']
    assert runtime_identity() == plan['runtime']
    populations = {}
    for task in plan['tasks']:
        path = ROOT / plan['dataset_files'][task]
        assert sha(path) == plan['dataset_sha256'][task]
        populations[task] = read_jsonl(path)
        assert [hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                for row in populations[task]] == plan['sample_ids'][task]
        assert bench.DATASET_TO_PROMPT[task] == plan['protocol']['prompt_templates'][task]
        assert bench.DATASET_TO_MAXGEN[task] == plan['protocol']['max_gen'][task]
    for path, digest in plan['reused_reference_records_sha256'].items():
        assert sha(ROOT / path) == digest
    record_dir = BASE / 'records' / f'shard_{args.shard}'
    record_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(record_dir / '.running.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f'{os.getpid()}\n'.encode())
    try:
        torch.manual_seed(0)
        with torch.inference_mode():
            model = frozen.AutoModelForCausalLM.from_pretrained(plan['model'],
                torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
            probe = model(torch.tensor([[1]], device='cuda'), use_cache=False,
                output_attentions=True, **bench.prefill_logit_kwargs(model, True))
            assert bool(torch.isfinite(probe.logits).all().item())
            del probe
            tokenizer = frozen.AutoTokenizer.from_pretrained(plan['model'])
            atomic_json(record_dir / 'runtime.json', {'plan_sha256': plan_hash,
                'runtime': runtime_identity(), 'shard': args.shard, 'shards': 2,
                'device': torch.cuda.get_device_name(),
                'device_capability': list(torch.cuda.get_device_capability()),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
            global_index = 0
            count = 0
            for task in plan['tasks']:
                for index, item in enumerate(populations[task]):
                    absolute = global_index
                    global_index += 1
                    if absolute % 2 != args.shard:
                        continue
                    sid = plan['sample_ids'][task][index]
                    prompt = plan['protocol']['prompt_templates'][task].format(**item)
                    if task not in plan['protocol']['no_chat_tasks']:
                        prompt = '[INST] ' + prompt + ' [/INST]'
                    ids = tokenizer(prompt, return_tensors='pt').input_ids
                    untruncated = ids.shape[1]
                    if untruncated > 31500:
                        ids = torch.cat([ids[:, :15750], ids[:, -15750:]], 1)
                    token_hash = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                    reference_path = FULL / 'records' / f'shard_{absolute % 2}' / f'{task}_{index:04d}_tiered_hot1024.json'
                    reference = json.loads(reference_path.read_text())
                    relative_reference = str(reference_path.relative_to(ROOT))
                    assert sha(reference_path) == plan['reused_reference_records_sha256'][relative_reference]
                    assert reference['execution_complete'] and reference['sample_id'] == sid
                    assert reference['input_ids_sha256'] == token_hash
                    ids = ids.to('cuda')
                    arms = plan['arms_by_task'][task]
                    shift = absolute % len(arms)
                    order = arms[shift:] + arms[:shift]
                    for arm in order:
                        path = record_dir / f'{task}_{index:04d}_{arm}.json'
                        if path.exists():
                            previous = json.loads(path.read_text())
                            assert previous['plan_sha256'] == plan_hash and previous['execution_complete']
                            assert previous['sample_id'] == sid and previous['input_ids_sha256'] == token_hash
                            assert previous['arm'] == arm
                            count += 1
                            continue
                        identity = {'plan_sha256': plan_hash, 'task': task,
                            'physical_file_row_index': index, 'global_row_index': absolute,
                            'shard': args.shard, 'shards': 2, 'arm': arm,
                            'fraction': plan['fractions'][arm], 'sample_id': sid,
                            'input_ids_sha256': token_hash,
                            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                            'untruncated_prompt_tokens': untruncated,
                            'dataset_length': item.get('length'),
                            'metric_function': plan['protocol']['metrics'][task],
                            'requested_backends': {'prefill': 'sdpa', 'decode': 'eager'},
                            'balanced_arm_order': order,
                            'reference_fraction075_record': relative_reference,
                            'reference_fraction075_record_sha256': plan['reused_reference_records_sha256'][relative_reference]}
                        try:
                            result = run_request(model, tokenizer, ids, task, item, plan['fractions'][arm])
                            assert result['generated_token_ids'][0] == reference['generated_token_ids'][0], 'Common full-prefill first-token invariant failed'
                            assert not path.exists()
                            atomic_json(path, {**identity, **result, 'execution_complete': True})
                            count += 1
                            print(task, index, arm, result['score'], flush=True)
                        except BaseException as error:
                            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                            atomic_json(BASE / 'errors' / f'shard_{args.shard}_{task}_{index}_{arm}_{stamp}.json',
                                {**identity, 'execution_complete': False,
                                 'error': f'{type(error).__name__}: {error}',
                                 'traceback': traceback.format_exc()})
                            raise
                        finally:
                            gc.collect()
                            torch.cuda.empty_cache()
                    del ids, reference
            atomic_json(record_dir / 'complete.json', {'plan_sha256': plan_hash,
                'execution_complete': True, 'total_population_rows': global_index,
                'new_request_records': count})
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    main()
