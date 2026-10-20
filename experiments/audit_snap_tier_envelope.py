"""Audit shared owned-KV and attended-width ceilings; actual usage may differ."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from audit_snap_tier_resources import snapshot
from direct_shared_pool import DirectScorePolicy
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy, _attn_impl, apply_tuple_kv_inplace, to_tuple_kv
from provenance import atomic_json, runtime_identity
from snapkv_adapter import SnapKVAdapter


# Frozen algebraic ceilings, not tuned to outputs: one Mistral grouped KV row
# costs 128 KiB/model; one upstream expanded SnapKV row costs 512 KiB/model.
# GPU: hot + cold + visible = 991 + 2082 + 1023 = 4096 grouped rows.
# CPU: full slot root + hot + visible = (991+1090+1)+991+1023 = 4096.
# Next single-token forward adds one row to the 1023-row visible boundary.
SETTINGS = {'snap': dict(prompt_rows=974, window=32, kernel=5),
            'gpu': dict(hot=991, cold=2082, exposed=32),
            'cpu_direct': dict(hot=991, cold=1090, exposed=32)}
KV_CEILING = 512 * 2**20
WIDTH_CEILING = 1024


@torch.inference_mode()
def run(model, ids, stream, arm):
    settings = SETTINGS[arm]
    policy = capture = past = None
    try:
        if arm == 'snap':
            past = SnapKVAdapter(model, capacity=settings['prompt_rows'],
                                 window=settings['window'], kernel=settings['kernel'])
        else:
            cls = TieredKVPolicy if arm == 'gpu' else DirectScorePolicy
            policy = cls(sram_budget=settings['hot'], stt_budget=settings['cold'],
                         stt_expose_quota=settings['exposed'], start_size=4,
                         recent_size=128, kv_dtype=torch.bfloat16,
                         sttram_bifurcation_frac=.75)
            policy.prompt_query_score_mode = 'chunked'
            policy.prompt_query_chunk = 128
            capture = PromptQueryCapture(model, policy)
        torch.cuda.reset_peak_memory_stats()
        with _attn_impl(model, 'sdpa'):
            out = model(ids, past_key_values=past, use_cache=True,
                        position_ids=torch.arange(ids.shape[1], device='cuda')[None])
        past = out.past_key_values
        if policy is not None:
            apply_tuple_kv_inplace(past, policy(to_tuple_kv(past)))
        if capture is not None:
            capture.close(); capture = None
        rows = [snapshot(past, policy, 0)]
        predictions = []
        for step, teacher in enumerate(stream):
            widths = [layer.keys.shape[-2] + 1 for layer in past.layers]
            assert max(widths) <= WIDTH_CEILING
            out = model(torch.tensor([[teacher]], device='cuda'), past_key_values=past,
                        use_cache=True, position_ids=torch.tensor([[ids.shape[1]+step]], device='cuda'),
                        output_attentions=policy is not None)
            predictions.append(out.logits[:, -1:].argmax(-1).item())
            past = out.past_key_values
            if policy is not None:
                tup = to_tuple_kv(past)
                policy.update_scores(tup, out.attentions)
                apply_tuple_kv_inplace(past, policy(tup))
            row = snapshot(past, policy, step+1)
            row['forward_attended_rows_by_layer'] = widths
            rows.append(row)
        for row in rows:
            assert sum(row['distinct_owned_kv_storage_bytes'].values()) <= KV_CEILING
        torch.cuda.synchronize()
        return dict(arm=arm, boundaries=rows, predicted_token_ids=predictions,
                    diagnostic_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    counters=policy._live_totals() if policy is not None else None,
                    h2d_payload_bytes=sum(getattr(c, 'host_read_bytes', 0) for c in policy.layer_caches.values()) if policy else 0,
                    d2h_payload_bytes=sum(getattr(c, 'host_write_bytes', 0) for c in policy.layer_caches.values()) if policy else 0)
    finally:
        if capture is not None:
            capture.close()
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
        raise FileExistsError(f'Preserve evidence: {args.output}')
    names = ['experiments/audit_snap_tier_envelope.py', 'experiments/audit_snap_tier_resources.py',
             'experiments/snapkv_adapter.py', 'sota/snapkv/snapkv/monkeypatch/snapkv_utils.py',
             'experiments/longbench_eval.py', 'experiments/direct_shared_pool.py',
             'experiments/batched_score_policy.py', 'experiments/cpu_batched_cache.py',
             'experiments/cpu_slot_cache.py', 'experiments/evaluate_query_selection.py',
             'src/tiered_kv_cache.py']
    report = dict(passed=False, runtime=runtime_identity(), settings=SETTINGS, rows=[],
                  owned_kv_ceiling_bytes=KV_CEILING, attended_rows_ceiling=WIDTH_CEILING,
                  source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names})
    try:
        prompt = Path('results/remediation_v3/latency_prompt.txt').read_text()
        teacher = Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        stream = json.loads(teacher.read_text())['config']['teacher_token_ids'][:50]
        assert len(stream) == 50
        report.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      teacher_source_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
                      teacher_token_ids=stream)
        torch.manual_seed(0)
        path = '../models/mistral-7b-instruct'
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                        device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(path)
        ids = tokenizer(prompt, return_tensors='pt').input_ids[:, :16384].to('cuda')
        assert ids.shape == (1, 16384)
        report.update(input_token_ids=ids.cpu().tolist()[0], gpu=torch.cuda.get_device_name())
        for arm in SETTINGS:
            result = run(model, ids, stream, arm)
            report['rows'].append(result)
            atomic_json(args.output, report)
            print(arm, result['boundaries'][-1]['distinct_owned_kv_storage_bytes'], flush=True)
        report.update(passed=True, shared_ceiling_verified=True, actual_resources_equal=False,
                      limitations='One fixed teacher stream and model; equal ceilings, unequal actual bytes/widths and cold capacities. Boundary KV ownership includes visible copies and complete pinned roots, excludes metadata, retained allocator blocks and transient staging. Observed CUDA peaks include diagnostic synchronization. No quality, latency, throughput, model-memory saving or physical-STT gate.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)


if __name__ == '__main__':
    main()
