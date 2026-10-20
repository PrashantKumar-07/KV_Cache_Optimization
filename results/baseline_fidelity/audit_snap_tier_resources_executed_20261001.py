"""Observe physical KV ownership and attended geometry at every decode boundary."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from batched_score_policy import ScoreBatchPolicy
from longbench_eval import TieredKVPolicy, _attn_impl, apply_tuple_kv_inplace, to_tuple_kv
from provenance import atomic_json, runtime_identity
from snapkv_adapter import SnapKVAdapter


def distinct_storage_bytes(tensors):
    """Count whole live storage allocations once, including capacity behind views."""
    seen, totals = set(), {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            totals[key[0]] = totals.get(key[0], 0) + storage.nbytes()
    return totals


def snapshot(past, policy, step):
    visible = [t for layer in past.layers for t in (layer.keys, layer.values)]
    tensors = list(visible)
    layers = []
    for i, layer in enumerate(past.layers):
        item = dict(layer=i, visible_rows=layer.keys.shape[-2], stored_visible_heads=layer.keys.shape[1])
        if policy is not None:
            cache = policy.layer_caches[i]
            hot = [cache.sram_k, cache.sram_v]
            tensors.extend(hot)
            if hasattr(cache, 'pool') and cache.pool is not None:
                tensors.append(cache.pool)
                cold_payload = cache.occupied_host_bytes
                cold_reserved = cache.reserved_host_bytes
            else:
                cold = [cache.stt_k, cache.stt_v]
                tensors.extend(cold)
                cold_payload = cold_reserved = sum(t.numel() * t.element_size() for t in cold)
            item.update(hot_rows=len(cache.sram_pos), cold_rows=len(cache.stt_pos),
                shadow_rows=int(cache.stt_shadow.sum().item()), hot_payload_bytes=sum(t.numel()*t.element_size() for t in hot),
                cold_occupied_payload_bytes=cold_payload, cold_reserved_payload_bytes=cold_reserved)
        layers.append(item)
    if policy is not None and getattr(policy, 'manager', None) is not None:
        assert not policy.manager.pending, 'Boundary still owns queued transfer tensors'
        assert all(c._prefetched is None for c in policy.layer_caches.values())
    return dict(step=step, layers=layers, distinct_owned_kv_storage_bytes=distinct_storage_bytes(tensors),
        visible_kv_tensor_payload_bytes=sum(t.numel()*t.element_size() for t in visible),
        process_gpu_allocated_bytes=torch.cuda.memory_allocated(),
        process_gpu_reserved_bytes=torch.cuda.memory_reserved())


@torch.inference_mode()
def run(model, ids, stream, arm):
    policy = None if arm == 'snap' else (TieredKVPolicy if arm == 'gpu' else ScoreBatchPolicy)(
        sram_budget=1024, stt_budget=2048, start_size=4, recent_size=128,
        stt_expose_quota=32, kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
    past = SnapKVAdapter(model, capacity=974, window=32, kernel=5) if arm == 'snap' else None
    try:
        torch.cuda.reset_peak_memory_stats()
        with _attn_impl(model, 'sdpa'):
            out = model(ids, past_key_values=past, use_cache=True,
                position_ids=torch.arange(ids.shape[1], device='cuda').unsqueeze(0))
        past = out.past_key_values
        if policy is not None:
            apply_tuple_kv_inplace(past, policy(to_tuple_kv(past)))
        rows = [snapshot(past, policy, 0)]
        predictions = []
        for step, teacher in enumerate(stream):
            out = model(torch.tensor([[teacher]], device='cuda'), past_key_values=past, use_cache=True,
                position_ids=torch.tensor([[ids.shape[1] + step]], device='cuda'),
                output_attentions=policy is not None)
            predictions.append(out.logits[:, -1:].argmax(-1).item())
            past = out.past_key_values
            if policy is not None:
                tup = to_tuple_kv(past)
                policy.update_scores(tup, out.attentions)
                apply_tuple_kv_inplace(past, policy(tup))
            rows.append(snapshot(past, policy, step + 1))
        torch.cuda.synchronize()
        return dict(arm=arm, boundaries=rows, predicted_token_ids=predictions,
            diagnostic_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        if policy is not None:
            policy.reset()
        elif past is not None:
            past.close()
        del past, policy
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    sources = ['experiments/audit_snap_tier_resources.py', 'experiments/snapkv_adapter.py',
        'sota/snapkv/snapkv/monkeypatch/snapkv_utils.py', 'experiments/longbench_eval.py',
        'src/tiered_kv_cache.py', 'experiments/batched_score_policy.py',
        'experiments/cpu_batched_cache.py', 'experiments/cpu_slot_cache.py']
    report = dict(passed=False, runtime=runtime_identity(), rows=[],
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    try:
        prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
        teacher = Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        stream = json.loads(teacher.read_text())['config']['teacher_token_ids'][:50]
        assert len(stream) == 50
        report.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            teacher_source_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(), teacher_token_ids=stream)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        ids = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids[:, :16384].to('cuda')
        assert ids.shape == (1, 16384)
        report.update(input_token_ids=ids.cpu().tolist()[0], gpu=torch.cuda.get_device_name())
        for arm in ('snap', 'gpu', 'cpu_scores'):
            report['rows'].append(run(model, ids, stream, arm))
            atomic_json(args.output, report)
            print(arm, report['rows'][-1]['boundaries'][-1]['distinct_owned_kv_storage_bytes'], flush=True)
        gpu, cpu = report['rows'][1:]
        assert gpu['predicted_token_ids'] == cpu['predicted_token_ids']
        geometry = lambda row: [{k: v for k, v in layer.items() if k != 'cold_reserved_payload_bytes'}
                               for layer in row['layers']]
        assert all(geometry(g) == geometry(c) for g, c in zip(gpu['boundaries'], cpu['boundaries']))
        report.update(passed=True, gpu_cpu_geometry_and_predictions_equal=True,
            resource_matched=False, scope='Fixed teacher trajectory, no quality or latency gate; unique KV storage includes physical reservations and visible cache copies, excludes metadata/model/logits. PyTorch GPU allocations include model/attention work; pinned allocator retained blocks and process RSS unmeasured.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
