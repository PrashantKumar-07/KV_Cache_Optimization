"""One diagnostic trace; unchanged fixed-teacher full-query path, no speed gate."""
import gc
import hashlib
import json
import runpy
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments'))
from compare_cpu_batched import digest
from evaluate_query_selection import PromptQueryCapture
from gpu_query_read_indices import GpuReadQueryPolicy
from latency_profile import profile_one
from profile_query_phases import ProfiledGpuQuerySelector, PHASE_AST
from provenance import atomic_json, runtime_identity

BASE = Path(__file__).resolve().parent
quiet = runpy.run_path(str(ROOT / 'results/hardware/pool_query_selector_v1/check_timing.py'))['quiet']
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def metric(row):
    return dict(name=row.key, count=row.count,
        cpu_elapsed_total_ms=row.cpu_time_total/1000,
        cpu_self_total_ms=row.self_cpu_time_total/1000,
        attributed_device_total_ms=row.device_time_total/1000,
        device_self_total_ms=row.self_device_time_total/1000)


@torch.inference_mode()
def main():
    output = BASE / 'profile_v4.json'
    assert not output.exists()
    plan = json.loads((BASE / 'plan.json').read_text())
    reference = json.loads((ROOT / plan['control']).read_text())
    for key in ('control', 'prompt', 'teachers'):
        assert sha(ROOT / plan[key]) == plan[key + '_sha256']
    report = dict(passed=False, plan_sha256=sha(BASE/'plan.json'), runtime=runtime_identity(),
                  quiet_observations=[], phase_ast=PHASE_AST)
    policy = capture = selector = model = None
    before = None
    def observe(stage):
        observation = dict(stage=stage, **quiet())
        report['quiet_observations'].append(observation)
        assert observation['quiet'], observation
    try:
        observe('before_model_load')
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        before = [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
        policy = GpuReadQueryPolicy(sram_budget=991, stt_budget=1090, stt_expose_quota=32,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)
        policy.prompt_query_score_mode='chunked'; policy.prompt_query_chunk=128
        capture = PromptQueryCapture(model, policy)
        selector = ProfiledGpuQuerySelector(model, policy)
        stream = json.loads((ROOT / plan['teachers']).read_text())['config']['teacher_token_ids']
        options = dict(ctx_len=16384, n_decode=50, n_warmup=0, n_repeat=1, device='cuda',
            prompt_text=(ROOT/plan['prompt']).read_text(), forced_token_ids=stream, prefill_last_logits=True)
        observe('before_warmup')
        warmup = profile_one(model, tokenizer, policy, **options)
        assert warmup['trials'][0]['predicted_token_ids'] == reference['predicted_token_ids']
        assert digest(policy.layer_caches) == reference['state_sha256']
        observe('before_trace')
        selector.selected_layers = selector.scanned_rows = 0
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as profiler:
            result = profile_one(model, tokenizer, policy, **options)
        observe('after_trace')
        assert result['trials'][0]['predicted_token_ids'] == reference['predicted_token_ids']
        assert digest(policy.layer_caches) == reference['state_sha256']
        assert policy._live_totals() == reference['counters']
        for key in ('host_read_bytes', 'host_write_bytes', 'host_initial_write_bytes'):
            assert sum(getattr(c,key) for c in policy.layer_caches.values()) == reference['prototype_counts'][key]
        assert selector.selected_layers == 1600
        assert policy.query_reader.metadata_h2d_bytes == 75333120
        assert policy.query_reader.refreshes == 51 and policy.query_reader.reads == 1600
        assert policy.manager.h2d_calls == 1701 and policy.manager.d2h_calls == 78
        rows = list(profiler.key_averages())
        phases = [metric(row) for row in rows if row.key.startswith('query_phase/')]
        report.update(phases=phases, fixed_query_state_counters_transfers_replayed=True,
            measured_selected_layers=selector.selected_layers)
        atomic_json(output, report)
        print('Raw phase observations:', phases, flush=True)
        assert len(phases) == 8 and all(row['count']==1600 for row in phases)
        assert {row['name'].split('/')[1] for row in phases} == set(PHASE_AST['phases'])
        trace = BASE/'trace_v4.json'
        assert not trace.exists()
        profiler.export_chrome_trace(str(trace))
        events = profiler.events()
        cuda_events = [event for event in events if event.device_type == torch.autograd.DeviceType.CUDA]
        report.update(passed=True, frozen_query_replay=True, predicted_token_ids=result['trials'][0]['predicted_token_ids'],
            state_sha256=digest(policy.layer_caches), logical_counters=policy._live_totals(),
            instrumented_result=result, selected_layers=selector.selected_layers, scanned_rows=selector.scanned_rows,
            native_h2d_bytes=sum(c.host_read_bytes for c in policy.layer_caches.values()),
            native_d2h_bytes=sum(c.host_write_bytes for c in policy.layer_caches.values()),
            native_h2d_calls=policy.manager.h2d_calls, native_d2h_calls=policy.manager.d2h_calls,
            index_metadata_h2d_bytes=policy.query_reader.metadata_h2d_bytes,
            phases=phases, operator_top_cpu_self=sorted([metric(row) for row in rows],key=lambda r:r['cpu_self_total_ms'],reverse=True)[:30],
            operator_top_device_self=sorted([metric(row) for row in rows],key=lambda r:r['device_self_total_ms'],reverse=True)[:30],
            profiler_cpu_events=len(events)-len(cuda_events), profiler_cuda_events=len(cuda_events),
            trace_sha256=sha(trace), trace_bytes=trace.stat().st_size,
            limits=plan['limits'], ordinary_adoption=False, full_goal_complete=False)
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        if selector: selector.close()
        if capture: capture.close()
        if policy:
            policy.reset()
            assert policy.query_reader.blocks is None and policy.query_reader.live is None
        if model and before is not None:
            report['own_hooks_removed'] = before == [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
            assert report['own_hooks_removed']
        report['index_metadata_released'] = policy is None or (policy.query_reader.blocks is None and policy.query_reader.live is None)
        policy = capture = selector = None
        gc.collect()
        try:
            if model:
                torch.cuda.synchronize()
                observe('after_cleanup')
        finally:
            atomic_json(output, report)
    for row in report['phases']:
        print(row, flush=True)
    print('Diagnostic fixed-query replay and hook/index cleanup passed', flush=True)

if __name__=='__main__': main()
