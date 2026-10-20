"""Prospective faithful Snap/native comparison; see immutable plan.json.

No GPU work occurs on import or --validate-only. Resource snapshots are a
separate, deliberately untimed request after caller cache commit/output release.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import tarfile
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
from reusable_exact_root import ExactRootOwner, ReusableExactPolicy
from exact_pinned_memory import stats as allocation_stats
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from snapkv_adapter import SnapKVAdapter
from longbench_eval import _attn_impl, prefill_logit_kwargs, to_tuple_kv, apply_tuple_kv_inplace
from latency_profile import cuda_sync_time
from audit_snap_tier_resources import distinct_storage_bytes
from audit_pinned_blocks import kernel as pinned_probe
from provenance import atomic_json, model_artifact_identity, runtime_identity


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(plan):
    assert sha(__file__) == plan['runner_sha256']
    assert sha(ROOT / plan['source_archive']) == plan['source_archive_sha256']
    frozen = {}
    with tarfile.open(ROOT / plan['source_archive'], 'r|gz') as archive:
        for member in archive:
            if member.isfile():
                frozen[member.name.removeprefix('./')] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert frozen == plan['source_sha256']
    assert all(sha(ROOT / name) == value for name, value in frozen.items())
    for name, value in plan['prerequisite_sha256'].items():
        assert sha(ROOT / name) == value, name
    for name in ('prompt', 'teacher_stream', 'quiet_helper'):
        assert sha(ROOT / plan[name]) == plan[name + '_sha256']
    assert runtime_identity() == plan['runtime']
    assert model_artifact_identity(plan['model']) == plan['model_identity']
    return {'source_files': len(frozen), 'archive_live_plan_match': True,
            'runtime_model_prerequisites_match': True}


def hooks(model):
    modules = list(model.modules())
    return [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in modules], [
        layer.self_attn.num_key_value_groups for layer in model.model.layers]


def counts(policy):
    names = ('cold_boundary_steps', 'cold_boundary_fallback_calls', 'cpu_append_calls',
             'cpu_fallback_steps', 'bulk_payload_steps', 'bulk_payload_fallback_steps',
             'bulk_descriptor_h2d_bytes', 'cold_boundary_descriptor_h2d_bytes',
             'controller_cpu_streak_array_peak')
    return {**{name: getattr(policy, name, 0) for name in names},
            'controller_address_array_peak': getattr(policy.manager, 'controller_address_array_peak', 0)}


def boundary(past, policy):
    """Called only after apply_tuple_kv_inplace and deletion of forward output."""
    torch.cuda.synchronize()
    visible = [(layer.keys, layer.values) for layer in past.layers]
    tensors = [tensor for pair in visible for tensor in pair]
    layers = []
    for index, (key, value) in enumerate(visible):
        row = {'layer': index, 'visible_rows': key.shape[-2], 'visible_heads': key.shape[-3]}
        if policy is not None:
            cache = policy.layer_caches[index]
            tensors.extend([cache.sram_k, cache.sram_v, cache.pool])
            row.update(hot_rows=len(cache.sram_pos), cold_total_rows=len(cache.stt_pos),
                       cold_live_rows=int((~cache.stt_shadow).sum().item()))
        layers.append(row)
    return {'layers': layers, 'deduplicated_named_KV_storage_bytes': distinct_storage_bytes(tensors),
            'visible_tensor_payload_bytes': sum(t.numel() * t.element_size() for pair in visible for t in pair),
            'gpu_allocated_bytes': torch.cuda.memory_allocated(), 'gpu_reserved_bytes': torch.cuda.memory_reserved(),
            'cpu_canonical_metadata_bytes': 0 if policy is None else policy.cpu_arena.owned_bytes}


@torch.inference_mode()
def profile(model, ids, forced, arm, policy=None, *, warmup=1, repeat=2, diagnostic=False):
    """Full-loop semantics mirror latency_profile.profile_one for native.

    Snap owns expanded heads and grows on decode as upstream specifies. Both
    include required policy work and host-visible argmax delivery. Native needs
    attention feedback; Snap does not request unused attention outputs.
    """
    ctx = ids.shape[1]
    options = prefill_logit_kwargs(model, True)
    trials, trace = [], []
    for trial in range(warmup + repeat):
        if policy is not None:
            policy.reset()
        cache = SnapKVAdapter(model, capacity=904, window=32, reuse_projection=True) if arm == 'snap' else None
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            def prefill():
                pos = torch.arange(ctx, device='cuda').unsqueeze(0)
                with _attn_impl(model, 'sdpa'):
                    forward_ms, out = cuda_sync_time(lambda: model(
                        ids, past_key_values=cache, use_cache=True, position_ids=pos, **options), 'cuda')
                past = out.past_key_values
                if policy is not None:
                    apply_tuple_kv_inplace(past, policy(to_tuple_kv(past)))
                assert torch.isfinite(out.logits).all().item(), 'Nonfinite prefill logits'
                prediction = out.logits[:, -1:].argmax(-1).item()
                return past, forced[:, :1], forward_ms, prediction

            ttft, (past, token, prefill_forward_ms, prefill_prediction) = cuda_sync_time(prefill, 'cuda')
            # The nested forward output has expired before this postcommit observer.
            prefill_alloc = torch.cuda.max_memory_allocated()
            prefill_reserved = torch.cuda.max_memory_reserved()
            if diagnostic:
                trace.append(boundary(past, policy))
            torch.cuda.reset_peak_memory_stats()
            steps, forwards, predictions = [], [], []
            for step in range(50):
                def decode():
                    pos = torch.tensor([[ctx + step]], device='cuda')
                    forward_ms, out = cuda_sync_time(lambda: model(
                        token, past_key_values=past, use_cache=True, position_ids=pos,
                        output_attentions=policy is not None), 'cuda')
                    raw = out.past_key_values
                    if policy is not None:
                        tup = to_tuple_kv(raw)
                        policy.update_scores(tup, out.attentions)
                        apply_tuple_kv_inplace(raw, policy(tup))
                    assert torch.isfinite(out.logits).all().item(), 'Nonfinite decode logits'
                    prediction = out.logits[:, -1:].argmax(-1).item()
                    return raw, forced[:, step + 1:step + 2], forward_ms, prediction
                elapsed, (past, token, forward_ms, prediction) = cuda_sync_time(decode, 'cuda')
                steps.append(elapsed)
                forwards.append(forward_ms)
                predictions.append(prediction)
                if diagnostic:
                    trace.append(boundary(past, policy))
            record = {'ttft_ms': ttft, 'step_ms': steps, 'forward_ms': forwards,
                      'prefill_forward_ms': prefill_forward_ms,
                      'prefill_prediction': prefill_prediction, 'predicted_token_ids': predictions,
                      'all_forward_logits_finite': True,
                      'prefill_peak_gpu_allocated_bytes': prefill_alloc,
                      'prefill_peak_gpu_reserved_bytes': prefill_reserved,
                      'decode_peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'decode_peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'final_visible_rows': [layer.keys.shape[-2] for layer in past.layers],
                      'warmup': trial < warmup}
            if policy is not None:
                record.update(counts=counts(policy), cpu_canonical_arrays_bytes=policy.cpu_arena.owned_bytes,
                              actual_pinned_pool=pinned_probe().block_info(policy.manager.pool),
                              pool_pointer=policy.manager.pool.data_ptr())
            # Keep warmup predictions/counters for correctness, exclude all warmup timings from summaries.
            trials.append(record)
            del past, token
        finally:
            if cache is not None:
                cache.close()
            del cache
    return {'timing_scope': 'diagnostic_invalid_timing' if diagnostic else 'end_to_end_model_policy_token',
            'trials': trials, **({'boundaries': trace} if diagnostic else {})}


def summarize(report, plan):
    summaries, scopes = {}, {}
    static = report['constant_model_storage_bytes']
    for context in plan['contexts']:
        arms = {}
        for arm in ('snap', 'native'):
            trials = [t for row in report['rows'] if row['context'] == context and row['arm'] == arm
                      for t in row['result']['trials'] if not t['warmup']]
            steps = [v for t in trials for v in t['step_ms']]
            assert len(steps) == 600
            arms[arm] = {'measured_decode_forwards': len(steps), 'median_itl_ms': float(np.median(steps)),
                         'p95_itl_ms': float(np.quantile(steps, .95)),
                         'median_ttft_ms': float(np.median([t['ttft_ms'] for t in trials]))}
            terms = []
            for trial in trials:
                item = {'decode_GPU_excluding_static_model': trial['decode_peak_gpu_allocated_bytes'] - static}
                if arm == 'native':
                    c = trial['counts']
                    item.update(observed_main_pinned_allocator_block=trial['actual_pinned_pool']['allocator_block_bytes'],
                                canonical_CPU_arrays=trial['cpu_canonical_arrays_bytes'],
                                CPU_streak_array_peak=c['controller_cpu_streak_array_peak'],
                                CPU_address_array_peak=c['controller_address_array_peak'],
                                registered_two_times_cumulative_descriptors=2 * (
                                    c['bulk_descriptor_h2d_bytes'] + c['cold_boundary_descriptor_h2d_bytes']))
                terms.append({'terms_bytes': item, 'scoped_bytes': sum(item.values())})
            scopes.setdefault(str(context), {})[arm] = terms
            arms[arm]['max_scoped_bytes'] = max(t['scoped_bytes'] for t in terms)
            for stage in ('prefill', 'decode'):
                for kind in ('allocated', 'reserved'):
                    field = f'{stage}_peak_gpu_{kind}_bytes'
                    arms[arm]['max_' + field] = max(t[field] for t in trials)
        ratios = {metric: arms['native'][metric] / arms['snap'][metric]
                  for metric in ('median_itl_ms', 'p95_itl_ms', 'median_ttft_ms')}
        guards = {metric: ratios[metric] <= bound for metric, bound in plan['timing_ratio_guards'].items()}
        guards.update({arm + '_scoped_budget': arms[arm]['max_scoped_bytes'] <= plan['scoped_budget_bytes']
                       for arm in arms})
        blocks = []
        rows = [row for row in report['rows'] if row['context'] == context]
        for block in range(3):
            medians = {arm: float(np.median([v for row in rows[4 * block:4 * block + 4] if row['arm'] == arm
                       for t in row['result']['trials'] if not t['warmup'] for v in t['step_ms']]))
                       for arm in ('snap', 'native')}
            blocks.append({'block': block, 'median_itl_ms': medians,
                           'native_snap_ratio': medians['native'] / medians['snap']})
        summaries[str(context)] = {'arms': arms, 'native_snap_ratios': ratios, 'guards': guards,
                                   'ABBA_blocks': blocks,
                                   'advancement_passed': all(guards.values())}
    pooled = {}
    for arm in ('snap', 'native'):
        trials = [t for row in report['rows'] if row['arm'] == arm for t in row['result']['trials'] if not t['warmup']]
        steps = [v for t in trials for v in t['step_ms']]
        pooled[arm] = {'measured_decode_forwards': len(steps), 'median_itl_ms': float(np.median(steps)),
                       'p95_itl_ms': float(np.quantile(steps, .95)),
                       'median_ttft_ms': float(np.median([t['ttft_ms'] for t in trials]))}
    pooled_ratios = {metric: pooled['native'][metric] / pooled['snap'][metric]
                     for metric in plan['timing_ratio_guards']}
    pooled_guards = {metric: pooled_ratios[metric] <= bound for metric, bound in plan['timing_ratio_guards'].items()}
    return {'per_context': summaries, 'pooled': {'arms': pooled, 'native_snap_ratios': pooled_ratios,
                                               'guards': pooled_guards},
            'scoped_memory_terms': scopes,
            'advancement_passed': all(v['advancement_passed'] for v in summaries.values()) and all(pooled_guards.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = json.loads((BASE / 'plan.json').read_text())
    validation = validate(plan)
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Launch only on physical GPU0'
    output = BASE / 'benchmark.json'
    assert not output.exists(), 'Preserve prior attempt; never overwrite measured evidence'
    spec = importlib.util.spec_from_file_location('quiet_observer', ROOT / plan['quiet_helper'])
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    report = {'execution_complete': False, 'pid': os.getpid(), 'plan_sha256': sha(BASE / 'plan.json'),
              'runtime': runtime_identity(), 'validation': validation, 'rows': [],
              'diagnostics': [], 'quiet_observations': []}
    owner = ExactRootOwner()

    def quiet(stage, **kw):
        observation = dict(stage=stage, **kw, **observer.quiet())
        report['quiet_observations'].append(observation)
        atomic_json(output, report)
        assert observation['quiet'], 'Unexpected GPU0 compute process'

    try:
        quiet('before_model_load')
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(plan['model'], torch_dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='eager').eval()
        assert model.config.model_type == 'mistral' and model.config.sliding_window is None
        # Transformers lazily installs persistent output-collector hooks on the
        # first attention-feedback forward. Initialize those symmetrically once
        # before ownership snapshots; adapter-owned hooks must still all close.
        with torch.inference_mode():
            framework_probe = model(torch.tensor([[1]], device='cuda'), use_cache=False,
                                    output_attentions=True, logits_to_keep=1)
            del framework_probe
        report['framework_output_hooks'] = {
            name: [getattr(h, '__module__', '') + ':' + getattr(h, '__qualname__', '')
                   for h in module._forward_hooks.values()]
            for name, module in model.named_modules() if module._forward_hooks}
        assert len(report['framework_output_hooks']) == 64
        assert all(len(v) == 1 and v[0].startswith('transformers.utils.output_capturing:')
                   for v in report['framework_output_hooks'].values())
        tokenizer = AutoTokenizer.from_pretrained(plan['model'])
        ids = tokenizer((ROOT / plan['prompt']).read_text(), truncation=False, return_tensors='pt').input_ids
        assert ids.shape[1] == plan['full_prompt_tokens']
        report['input_ids_sha256'] = {str(ctx): hashlib.sha256(ids[:, :ctx].numpy().tobytes()).hexdigest()
                                     for ctx in plan['contexts']}
        assert report['input_ids_sha256'] == plan['input_ids_sha256']
        forced = torch.tensor(json.loads((ROOT / plan['teacher_stream']).read_text())['config']['teacher_token_ids'][:51],
                              device='cuda').view(1, -1)
        assert forced.shape == (1, 51)
        static = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                  for t in [*model.parameters(), *model.buffers()] if t.device.type == 'cuda'}
        report['constant_model_storage_bytes'] = sum(static.values())
        schedule = [(True, 16384, arm) for arm in ('snap', 'native')] + [
            (False, ctx, arm) for ctx in plan['contexts'] for arm in plan['order']]
        reference = {}
        for index, (diagnostic, ctx, arm) in enumerate(schedule):
            quiet('before_diagnostic' if diagnostic else 'before_row', context=ctx, arm=arm)
            before = hooks(model)
            request_start = time.perf_counter()
            with ExitStack() as cleanup:
                policy = capture = lease = None
                if arm == 'native':
                    policy = ReusableExactPolicy(root_owner=owner, sram_budget=1024, stt_budget=2048,
                        stt_expose_quota=32, start_size=4, recent_size=128,
                        kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
                    cleanup.callback(policy.reset)
                    capture = PromptQueryCapture(model, policy)
                    cleanup.callback(capture.close)
                    lease = SharedAppendLease(model, policy)
                    cleanup.callback(lease.close)
                result = profile(model, ids[:, :ctx].to('cuda'), forced, arm, policy,
                                 warmup=0 if diagnostic else 1, repeat=1 if diagnostic else 2,
                                 diagnostic=diagnostic)
                row = {'index': index, 'context': ctx, 'arm': arm, 'result': result}
                for trial in result['trials']:
                    key = (ctx, arm)
                    prediction = [trial['prefill_prediction']] + trial['predicted_token_ids']
                    reference.setdefault(key, prediction)
                    assert reference[key] == prediction, 'Nonrepeatable arm/context predictions'
                    expected = plan['reference_predictions'].get(str(ctx), {}).get(arm)
                    if expected is not None:
                        assert trial['predicted_token_ids'] == expected, 'Mismatch against frozen fullmodel reference'
            row['hooks_closed'] = hooks(model) == before
            if diagnostic:
                row['invalid_timing_full_instrumented_request_wall_ms'] = (time.perf_counter() - request_start) * 1000
            row['lease_closed'] = lease is None or (not lease.handles and not lease.pending)
            assert row['hooks_closed'] and row['lease_closed']
            del policy, capture, lease
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            assert owner.token is None
            row.update(exact_allocation_stats_after_cleanup=allocation_stats(), owner_leases=owner.leases,
                       owner_allocation_ms=owner.allocation_ms,
                       retained_idle_pointer=None if owner.root is None else owner.root.data_ptr(),
                       experiment_retained_idle_root=None if owner.root is None else pinned_probe().block_info(owner.root))
            (report['diagnostics'] if diagnostic else report['rows']).append(row)
            quiet('after_diagnostic' if diagnostic else 'after_row', context=ctx, arm=arm)
            print('diagnostic' if diagnostic else 'timed', ctx, arm, flush=True)
        assert owner.leases == plan['expected_native_leases']
        owner.close()
        gc.collect()
        torch.cuda.synchronize()
        report['final_exact_allocation_stats'] = allocation_stats()
        assert allocation_stats()['live_allocations'] == 0
        report['summary'] = summarize(report, plan)
        report['execution_complete'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if not owner.closed and owner.token is None:
            owner.close()
        report['owner_closed'] = owner.closed and owner.root is None and owner.token is None
        report['final_exact_allocation_stats'] = allocation_stats()
        atomic_json(output, report)


if __name__ == '__main__':
    main()
