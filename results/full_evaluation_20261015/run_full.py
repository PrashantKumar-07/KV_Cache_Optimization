"""Frozen, complete-file LongBench quality evaluation with per-request resume.

Quality and logical modeled costs only: concurrent wall time is not a latency
benchmark. A completed record is never overwritten; failed attempts stay separate.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import longbench_eval as bench
from evaluate_query_selection import PromptQueryCapture
from snapkv_adapter import SnapKVAdapter
from provenance import atomic_json, model_artifact_identity, runtime_identity, read_jsonl


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hook_state(model):
    return ([ (len(m._forward_pre_hooks), len(m._forward_hooks)) for m in model.modules()],
            [l.self_attn.num_key_value_groups for l in model.model.layers])


def policy_for(arm):
    if arm == 'full':
        return bench.FullCachePolicy()
    if arm == 'streamingllm':
        return bench.StreamingLLMPolicy(start_size=4, recent_size=1020)
    if arm == 'h2o_adapted':
        return bench.H2OPolicy(budget=1024, start_size=4, recent_size=128)
    if arm == 'snapkv_adapted':
        return bench.SnapKVPolicy(budget=1024, start_size=4, recent_size=128)
    if arm in ('tiered_hot1024', 'tiered_hot992'):
        return bench.TieredKVPolicy(
            sram_budget=1024 if arm == 'tiered_hot1024' else 992,
            stt_budget=2048, stt_expose_quota=32, start_size=4,
            recent_size=128, kv_dtype=torch.bfloat16,
            sttram_bifurcation_frac=.75)
    if arm == 'snapkv_faithful':
        return None
    raise ValueError(arm)


def geometry(past, policy):
    """Physical tensor/storage accounting, not a whole-process memory claim."""
    layers = []
    storages = {}
    def tensor_info(tensor):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storages[key] = storage.nbytes()
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'device': str(tensor.device), 'logical_bytes': tensor.numel() * tensor.element_size(),
                'backing_storage_bytes': storage.nbytes()}
    for index, (k, v) in enumerate(bench.to_tuple_kv(past)):
        layer = {'layer': index, 'visible_k': tensor_info(k), 'visible_v': tensor_info(v)}
        if policy is not None and hasattr(policy, 'layer_caches') and index in policy.layer_caches:
            cache = policy.layer_caches[index]
            layer['tier_payloads'] = {name: tensor_info(getattr(cache, name))
                                     for name in ('sram_k', 'sram_v', 'stt_k', 'stt_v')}
            layer['cold_live_rows'] = int((~cache.stt_shadow).sum().item())
            layer['cold_shadow_rows'] = int(cache.stt_shadow.sum().item())
        layers.append(layer)
    return {'layers': layers, 'unique_payload_backing_bytes': sum(storages.values()),
            'scope': 'visible KV and tier hot/cold payload tensor storages; excludes metadata, weights, activations, allocator reserve'}


def run_request(model, tokenizer, ids, task, item, arm):
    before = hook_state(model)
    policy = policy_for(arm)
    n = ids.shape[1]
    generated = []
    widths = []
    baseline_lat = baseline_eng = 0.0
    started = time.monotonic()
    try:
        with ExitStack() as cleanup:
            if policy is not None:
                cleanup.callback(policy.reset)
                if arm.startswith('tiered_'):
                    capture = PromptQueryCapture(model, policy)
                    cleanup.callback(capture.close)
                with bench._attn_impl(model, 'sdpa'):
                    out = model(ids, use_cache=True,
                                position_ids=torch.arange(n, device=ids.device).unsqueeze(0),
                                **bench.prefill_logit_kwargs(model, True))
                past = out.past_key_values
                bench.apply_tuple_kv_inplace(past, policy(bench.to_tuple_kv(past)))
            else:
                past = SnapKVAdapter(model, 1024, 32, reuse_projection=True)
                cleanup.callback(past.close)
                with bench._attn_impl(model, 'sdpa'):
                    out = model(ids, past_key_values=past, use_cache=True,
                                position_ids=torch.arange(n, device=ids.device).unsqueeze(0),
                                **bench.prefill_logit_kwargs(model, True))
            initial = geometry(past, policy)
            for step in range(bench.DATASET_TO_MAXGEN[task]):
                if not bool(torch.isfinite(out.logits).all().item()):
                    raise FloatingPointError('Nonfinite model logits')
                token = out.logits[:, -1:].argmax(-1)
                tid = token.item()
                generated.append(tid)
                del out
                if tid == tokenizer.eos_token_id or step + 1 == bench.DATASET_TO_MAXGEN[task]:
                    break
                visible = bench.to_tuple_kv(past)
                widths.append([int(k.shape[2]) + 1 for k, _ in visible])
                if policy is None or hasattr(policy, '_modeled_latency_us'):
                    for k, _ in visible:
                        lat, eng = bench._BASELINE_COST_MODEL.baseline_attention_cost(
                            k.shape[2] + 1, k.shape[1], k.shape[3])
                        baseline_lat += lat
                        baseline_eng += eng
                        if policy is not None:
                            policy._modeled_latency_us += lat
                            policy._modeled_energy_nj += eng
                    if policy is not None:
                        policy._modeled_steps += 1
                del visible
                with bench._attn_impl(model, 'eager'):
                    out = model(token, past_key_values=past, use_cache=True,
                                position_ids=torch.tensor([[n + step]], device=ids.device),
                                output_attentions=True, **bench.prefill_logit_kwargs(model, True))
                if policy is not None:
                    raw = out.past_key_values
                    tup = bench.to_tuple_kv(raw)
                    update = getattr(policy, 'update_scores', None)
                    if update is not None:
                        update(tup, out.attentions)
                    bench.apply_tuple_kv_inplace(raw, policy(tup))
                    past = raw
                    del raw, tup
            final = geometry(past, policy)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            scored = text.lstrip('\n').split('\n')[0] if task in ('triviaqa', 'trec', 'samsum', 'lsht') else text
            score = 100 * max(bench.DATASET_TO_METRIC[task](
                scored, answer, all_classes=item.get('all_classes', [])) for answer in item['answers'])
            values = [width for row in widths for width in row]
            result = {'prediction': text, 'scored_prediction': scored, 'score': score,
                      'answers': item['answers'], 'generated_token_ids': generated,
                      'prompt_tokens': n, 'generated_tokens': len(generated),
                      'all_model_logits_finite': True,
                      'geometry_after_prefill_before_first_decode': initial,
                      'geometry_after_last_model_call_and_policy_commit': final,
                      'decode_attended_width': {'layer_calls': len(values),
                          'min': min(values) if values else None,
                          'max': max(values) if values else None, 'sum': sum(values),
                          'per_layer_sum': [sum(row[i] for row in widths) for i in range(len(widths[0]))] if widths else []},
                      'cache_stats': policy.stats() if policy is not None else {
                          'decode_steps': len(widths), 'modeled_latency_us': baseline_lat,
                          'modeled_energy_nj': baseline_eng},
                      'descriptive_elapsed_seconds_not_latency_benchmark': time.monotonic() - started}
            del past, token
        if hook_state(model) != before:
            raise RuntimeError('Hooks or KV head groups were not restored')
        result['hooks_and_groups_restored'] = True
        return result
    finally:
        if policy is not None:
            policy.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard index')
    plan_path = BASE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    assert sha(BASE / 'partitions.json') == plan['partition_sha256'], 'Partition hash mismatch'
    assert sha(__file__) == plan['runner_sha256'], 'Runner hash mismatch'
    assert all(sha(ROOT / path) == digest for path, digest in plan['source_sha256'].items()), 'Source hash mismatch'
    assert model_artifact_identity(plan['model']) == plan['model_identity'], 'Model identity mismatch'
    assert runtime_identity() == plan['runtime'], 'Runtime identity mismatch'
    for task, path in plan['dataset_files'].items():
        assert sha(ROOT / path) == plan['dataset_sha256'][task], 'Dataset hash mismatch'
    populations = {}
    for task in plan['tasks']:
        populations[task] = read_jsonl(ROOT / plan['dataset_files'][task])
        actual_ids = [hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                      for item in populations[task]]
        assert len(populations[task]) == plan['counts'][task]
        assert actual_ids == plan['sample_ids'][task], 'Population identity/order mismatch'
    assert len(plan['arms']) == 7 and len(set(plan['arms'])) == 7
    if 'shards' in plan:
        assert args.shards == plan['shards']
    protocol = plan['protocol']
    max_ctx = protocol.get('max_ctx', 31500)
    assert max_ctx == 31500
    assert protocol['seed'] == 0 and protocol['dtype'] == 'bfloat16'
    assert protocol['prefill_backend'] == 'sdpa' and protocol['decode_backend'] == 'eager'
    for task in plan['tasks']:
        assert protocol['max_gen'][task] == bench.DATASET_TO_MAXGEN[task]
        assert protocol['prompt_templates'][task] == bench.DATASET_TO_PROMPT[task]
        assert protocol['metrics'][task] == bench.DATASET_TO_METRIC[task].__name__
    record_dir = BASE / 'records' / f'shard_{args.shard}'
    record_dir.mkdir(parents=True, exist_ok=True)
    lock = record_dir / '.running.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f'{os.getpid()}\n'.encode())
    try:
        torch.manual_seed(0)
        with torch.inference_mode():
            model = AutoModelForCausalLM.from_pretrained(
                plan['model'], torch_dtype=torch.bfloat16, device_map='cuda',
                attn_implementation='eager').eval()
            probe = model(torch.tensor([[1]], device='cuda'), use_cache=False,
                          output_attentions=True, **bench.prefill_logit_kwargs(model, True))
            assert bool(torch.isfinite(probe.logits).all().item())
            del probe
            tokenizer = AutoTokenizer.from_pretrained(plan['model'])
            atomic_json(record_dir / 'runtime.json', {'plan_sha256': plan_hash,
                'runtime': runtime_identity(), 'device': torch.cuda.get_device_name(),
                'device_capability': list(torch.cuda.get_device_capability()),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'shard': args.shard, 'shards': args.shards})
            global_index = 0
            for task in plan['tasks']:
                rows = populations[task]
                assert len(rows) == plan['counts'][task]
                for index, item in enumerate(rows):
                    assigned_index = global_index
                    global_index += 1
                    if assigned_index % args.shards != args.shard:
                        continue
                    sid = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                    assert sid == plan['sample_ids'][task][index]
                    prompt = bench.DATASET_TO_PROMPT[task].format(**item)
                    if task not in bench.NO_CHAT_TASKS:
                        prompt = '[INST] ' + prompt + ' [/INST]'
                    ids = tokenizer(prompt, return_tensors='pt').input_ids
                    untruncated = ids.shape[1]
                    if untruncated > max_ctx:
                        ids = torch.cat([ids[:, :max_ctx // 2], ids[:, -max_ctx // 2:]], 1)
                    tokenhash = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                    ids = ids.to('cuda')
                    shift = assigned_index % len(plan['arms'])
                    order = plan['arms'][shift:] + plan['arms'][:shift]
                    for arm in order:
                        path = record_dir / f'{task}_{index:04d}_{arm}.json'
                        if path.exists():
                            previous = json.loads(path.read_text())
                            assert previous['plan_sha256'] == plan_hash
                            assert previous['sample_id'] == sid and previous['input_ids_sha256'] == tokenhash
                            assert previous['arm'] == arm and previous['execution_complete']
                            continue
                        identity = {'plan_sha256': plan_hash, 'task': task,
                            'physical_file_row_index': index, 'global_row_index': assigned_index,
                            'shard': args.shard, 'shards': args.shards, 'sample_id': sid,
                            'input_ids_sha256': tokenhash, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                            'untruncated_prompt_tokens': untruncated, 'arm': arm,
                            'dataset_length': item.get('length'),
                            'metric_function': bench.DATASET_TO_METRIC[task].__name__,
                            'requested_backends': {'prefill': 'sdpa', 'decode': 'eager'},
                            'balanced_arm_order': order}
                        try:
                            result = run_request(model, tokenizer, ids, task, item, arm)
                            for existing_path in record_dir.glob(f'{task}_{index:04d}_*.json'):
                                existing = json.loads(existing_path.read_text())
                                assert existing['plan_sha256'] == plan_hash
                                assert existing['generated_token_ids'][0] == result['generated_token_ids'][0], 'Uncompressed prefill first-token invariant failed'
                            assert not path.exists(), 'Completed record appeared concurrently'
                            atomic_json(path, {**identity, **result, 'execution_complete': True})
                            print(task, index, arm, result['score'], flush=True)
                        except BaseException as error:
                            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                            atomic_json(BASE / 'errors' / f'shard_{args.shard}_{task}_{index}_{arm}_{stamp}.json',
                                {**identity, 'execution_complete': False,
                                 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
                            raise
                        finally:
                            gc.collect()
                            torch.cuda.empty_cache()
                    del ids
            atomic_json(record_dir / 'complete.json', {'plan_sha256': plan_hash,
                'execution_complete': True, 'total_population_rows': global_index})
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    main()
