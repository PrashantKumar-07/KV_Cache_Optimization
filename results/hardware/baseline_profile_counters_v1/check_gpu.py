"""Five-policy fixed-stream schema checks and frozen/current bulk parity."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import latency_profile as profile
import compare_cpu_batched as trial
from compare_bulk_promotion import RecordedBulkPolicy
from longbench_eval import FullCachePolicy, StreamingLLMPolicy, H2OPolicy, SnapKVPolicy, TieredKVPolicy


def policy_for(name):
    return {'full': lambda: FullCachePolicy(),
        'streaming': lambda: StreamingLLMPolicy(4, 1020),
        'h2o': lambda: H2OPolicy(1024, 4, 128),
        'snapkv': lambda: SnapKVPolicy(1024, 4, 128),
        'tiered': lambda: TieredKVPolicy(sram_budget=1024, stt_budget=2048,
            start_size=4, recent_size=128, kv_dtype=torch.bfloat16)}[name]()


@torch.inference_mode()
def main():
    output = BASE / 'gpu.json'
    if output.exists():
        raise FileExistsError(output)
    report = dict(passed=False, rows=[], bulk_rows=[], timing_valid=False,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path('experiments').glob('*.py'), *Path('src').glob('*.py')]})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',
        torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
    prompt_path = ROOT / 'results/remediation_v3/latency_prompt.txt'
    teacher_path = ROOT / 'results/shadow_ablation_full/fixed_stream_20260929.json'
    prompt = prompt_path.read_text(); teacher = json.loads(teacher_path.read_text())['config']['teacher_token_ids']
    report.update(prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        teacher_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(), teacher_tokens=teacher[:13])
    events = []; handles = []

    def before(module, positional, options):
        ids = positional[0]
        events.append(dict(input_shape=list(ids.shape),
            token=ids.item() if ids.numel() == 1 else None,
            positions=options['position_ids'].tolist() if ids.numel() == 1 else None,
            last_logits_requested=options.get('logits_to_keep', 0)))

    def after(module, positional, options, out):
        events[-1].update(logit_shape=list(out.logits.shape),
            predicted=out.logits[:, -1:].argmax(-1).item())

    try:
        handles = [model.register_forward_pre_hook(before, with_kwargs=True),
            model.register_forward_hook(after, with_kwargs=True)]
        for name in ('full', 'streaming', 'h2o', 'snapkv', 'tiered'):
            predictions = []
            for last in (False, True):
                events.clear(); policy = policy_for(name)
                try:
                    result = profile.profile_one(model, tokenizer, policy, 4096,
                        n_decode=12, n_warmup=1, n_repeat=2, prompt_text=prompt,
                        forced_token_ids=teacher[:13], prefill_last_logits=last)
                finally:
                    policy.reset()
                assert len(events) == 39
                for start in (0, 13, 26):
                    assert events[start]['input_shape'] == [1, 4096]
                    assert events[start]['logit_shape'] == [1, 1 if last else 4096, 32000]
                    assert events[start]['last_logits_requested'] == (1 if last else 0)
                    for step, event in enumerate(events[start + 1:start + 13]):
                        assert event['token'] == teacher[step]
                        assert event['positions'] == [[4096 + step]]
                        assert event['logit_shape'] == [1, 1, 32000] and event['last_logits_requested'] == 0
                assert len(result['trials']) == 2
                for measured in result['trials']:
                    assert measured['decode_steps'] == 12
                    assert len(measured['predicted_token_ids']) == len(measured['step_ms']) == 12
                    for key in ('paid_writes', 'writes_saved', 'demoted', 'promoted'):
                        assert (key in measured) == (name == 'tiered')
                predictions.append([event['predicted'] for event in events])
                report['rows'].append(dict(policy=name, last_logits=last, events=list(events), result=result))
                print(name, last, 'counter schema and stream verified', flush=True)
            assert predictions[0] == predictions[1], name
        for handle in handles:
            handle.remove()
        handles.clear()
        functions = [node for node in ast.parse((BASE / 'latency_profile.py.before').read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == 'profile_one']
        assert len(functions) == 1
        namespace = profile.__dict__.copy()
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'archived_profiler', 'exec'), namespace)
        saved_profile, saved_policy = trial.profile_one, trial.FusedBoundedPolicy
        trial.FusedBoundedPolicy = RecordedBulkPolicy
        try:
            for arm, function in (('before', namespace['profile_one']), ('after', profile.profile_one)):
                trial.profile_one = function
                row = trial.measure(model, tokenizer, prompt, teacher, 4096, 12,
                    'fused_bounded_scores', 1, True, real_query_prefill=True, hot=991, cold=1090)
                report['bulk_rows'].append(dict(arm=arm, **row))
                print('ordinary bulk', arm, 'finished', flush=True)
            trial.check(*report['bulk_rows'])
            for field in ('reserved_host_bytes', 'native_d2h_calls', 'native_h2d_calls', 'per_layer_initial_d2h_calls'):
                assert report['bulk_rows'][0][field] == report['bulk_rows'][1][field]
        finally:
            trial.profile_one, trial.FusedBoundedPolicy = saved_profile, saved_policy
        report.update(passed=True, hook_cleanup_verified=True,
            limits='One local Mistral/seed,4K context/12decode. Primitive hooks invalidate timing. Different policy budgets; schema/input-stream and exact bulk preservation only, no fair-resource/quality/speed/physicalSTT claim.')
    finally:
        for handle in handles:
            handle.remove()
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
