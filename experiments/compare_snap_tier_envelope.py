"""Frozen same-ceiling SnapKV/direct-CPU TieredKV paired systems diagnostic."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from audit_snap_tier_envelope import SETTINGS, KV_CEILING, WIDTH_CEILING
from audit_snap_tier_resources import snapshot
from direct_shared_pool import DirectScorePolicy
from fused_bounded_pool import FusedBoundedPoolPolicy
from evaluate_query_selection import PromptQueryCapture
from latency_profile import cuda_sync_time
from longbench_eval import _attn_impl, apply_tuple_kv_inplace, to_tuple_kv
from provenance import atomic_json, runtime_identity
from snapkv_adapter import SnapKVAdapter


@torch.inference_mode()
def measure(model, ids, stream, arm, *, cpu_fused_bounded=False):
    trials = []
    for trial in range(2):  # One warm-up and one recorded trial in every arm.
        policy = capture = past = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if arm == 'snap':
                past = SnapKVAdapter(model, capacity=974, window=32, kernel=5)
            else:
                cfg = SETTINGS[arm]
                policy_cls = FusedBoundedPoolPolicy if cpu_fused_bounded else DirectScorePolicy
                policy = policy_cls(sram_budget=cfg['hot'], stt_budget=cfg['cold'],
                    stt_expose_quota=cfg['exposed'], start_size=4, recent_size=128,
                    kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
                policy.prompt_query_score_mode = 'chunked'; policy.prompt_query_chunk = 128
                capture = PromptQueryCapture(model, policy)
            def prefill():
                with _attn_impl(model, 'sdpa'):
                    _, out = cuda_sync_time(lambda:model(ids, past_key_values=past, use_cache=True,
                        position_ids=torch.arange(ids.shape[1], device='cuda')[None]), 'cuda')
                raw = out.past_key_values
                if policy is not None:
                    apply_tuple_kv_inplace(raw, policy(to_tuple_kv(raw)))
                out.logits[:, -1:].argmax(-1).item()
                return raw
            ttft, past = cuda_sync_time(prefill, 'cuda')
            if capture is not None:
                capture.close(); capture = None
            prefill_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            steps, forwards, predictions = [], [], []
            for step in range(50):
                def decode():
                    position = torch.tensor([[ids.shape[1]+step]], device='cuda')
                    token = stream[:, step:step+1]
                    forward, out = cuda_sync_time(lambda:model(token, past_key_values=past,
                        use_cache=True, position_ids=position, output_attentions=policy is not None), 'cuda')
                    raw = out.past_key_values
                    if policy is not None:
                        tup = to_tuple_kv(raw)
                        policy.update_scores(tup, out.attentions)
                        apply_tuple_kv_inplace(raw, policy(tup))
                    predicted = out.logits[:, -1:].argmax(-1).item()
                    return raw, forward, predicted
                elapsed, (past, forward, predicted) = cuda_sync_time(decode, 'cuda')
                steps.append(elapsed); forwards.append(forward); predictions.append(predicted)
            decode_peak = torch.cuda.max_memory_allocated()
            final = snapshot(past, policy, 50)  # Diagnostic synchronization outside timing.
            assert sum(final['distinct_owned_kv_storage_bytes'].values()) <= KV_CEILING
            assert max(x['visible_rows'] for x in final['layers']) <= WIDTH_CEILING
            if trial:
                trials.append(dict(ttft_ms=ttft, step_ms=steps, forward_ms=forwards,
                    predicted_token_ids=predictions, prefill_peak_gpu_bytes=prefill_peak,
                    decode_peak_gpu_bytes=decode_peak, final_boundary=final,
                    counters=policy._live_totals() if policy is not None else None,
                    h2d_payload_bytes=sum(getattr(c,'host_read_bytes',0) for c in policy.layer_caches.values()) if policy else 0,
                    d2h_payload_bytes=sum(getattr(c,'host_write_bytes',0) for c in policy.layer_caches.values()) if policy else 0))
        finally:
            if capture is not None: capture.close()
            if policy is not None: policy.reset()
            elif past is not None: past.close()
            del past, policy
            gc.collect(); torch.cuda.empty_cache()
    return trials[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-fused-bounded',action='store_true',
                        help='Use the verified composed CPU backend at the frozen capacities')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f'Preserve evidence: {args.output}')
    sources = list(Path('experiments').glob('*.py')) + list(Path('src').glob('*.py')) + [
        Path('experiments/native_hot_compact.cpp'),Path('experiments/native_hot_compact.cu')]
    report = dict(passed=False, runtime=runtime_identity(), rows=[], settings=SETTINGS,
        owned_kv_ceiling_bytes=KV_CEILING, attended_row_ceiling=WIDTH_CEILING,
        cpu_fused_bounded=args.cpu_fused_bounded,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        timing_scope='Full model, required policy work and host-visible token, identical outer/inner synchronization; boundary audit excluded')
    try:
        if args.cpu_fused_bounded:
            from fused_hot_compaction import kernel
            report['fused_binary_sha256']=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest()
        torch.manual_seed(0)
        path = '../models/mistral-7b-instruct'
        model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,
                device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(path)
        prompt=Path('results/remediation_v3/latency_prompt.txt').read_text()
        teachers=Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        stream=json.loads(teachers.read_text())['config']['teacher_token_ids'][:50]
        assert len(stream)==50
        ids=tokenizer(prompt,return_tensors='pt').input_ids[:,:16384].to('cuda')
        stream_tensor=torch.tensor([stream],device='cuda')
        report.update(model=path,seed=0,context=16384,decode=50,warmup=1,repeat=1,
            input_token_ids=ids.cpu().tolist()[0],teacher_token_ids=stream,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            teacher_source_sha256=hashlib.sha256(teachers.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name())
        reference = {}
        for block in range(3):
            for arm in ('snap','cpu_direct','cpu_direct','snap'):
                result=measure(model,ids,stream_tensor,arm,cpu_fused_bounded=args.cpu_fused_bounded)
                stable={name:result[name] for name in ('predicted_token_ids','counters','h2d_payload_bytes','d2h_payload_bytes')}
                geometry=result['final_boundary']
                stable['final_storage']=geometry['distinct_owned_kv_storage_bytes']
                stable['final_layers']=geometry['layers']
                if arm in reference: assert stable==reference[arm], f'{arm} repeatability'
                else: reference[arm]=stable
                report['rows'].append(dict(block=block,arm=arm,**result))
                atomic_json(args.output,report)
                print(block,arm,statistics.median(result['step_ms']),flush=True)
        report.update(passed=True,limitations='One fixed teacher trajectory/model/seed and equal boundary resource ceilings; actual bytes, widths, cold retention, CPU vs GPU tier placement and required attention-weight work differ. Transient live storage and allocator blocks are excluded from the boundary ceiling. No joint quality, physical STT, exact-resource equality, serving or research-completion claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        atomic_json(args.output,report)


if __name__=='__main__': main()
