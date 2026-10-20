"""Independent complete-run verification; --progress exposes counts only.

No aggregate score is calculated until every expected request and both shard
completion records exist. Raw records are read only. No CUDA is used.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def expected_records(plan):
    expected = {}
    global_index = 0
    for task in plan['tasks']:
        for index, sid in enumerate(plan['sample_ids'][task]):
            shard = global_index % 2
            for arm in plan['arms']:
                path = BASE / 'records' / f'shard_{shard}' / f'{task}_{index:04d}_{arm}.json'
                expected[path] = (task, index, sid, global_index, shard, arm)
            global_index += 1
    assert len(expected) == plan['total_predictions'] == 10318
    assert global_index == plan['total_rows'] == 1474
    return expected


def completion_gate(plan, expected):
    present = set((BASE / 'records').glob('shard_*/*_*.json'))
    # runtime/complete metadata have no underscore; every other JSON must be expected.
    unexpected = present - set(expected)
    assert not unexpected, f'Unexpected request paths: {sorted(map(str, unexpected))[:5]}'
    counts = Counter((expected[p][0], expected[p][-1]) for p in present)
    metadata = [BASE / 'records' / f'shard_{s}' / 'complete.json' for s in range(2)]
    complete = len(present) == len(expected) and all(p.exists() for p in metadata)
    return complete, {'completed_request_files': len(present),
        'expected_request_files': len(expected), 'population_rows': plan['total_rows'],
        'shards_marked_complete': [s for s, p in enumerate(metadata) if p.exists()],
        'per_task_arm_counts': {t: {a: counts[(t, a)] for a in plan['arms']} for t in plan['tasks']},
        'all_expected_files_present': len(present) == len(expected),
        'complete': complete, 'scope': 'File counts only; no predictions or scores opened or aggregated'}


def reference_metrics():
    from rouge import Rouge
    reference = ROOT / 'sota/snapkv/experiments/LongBench/metrics.py'
    names = {'normalize_answer', 'f1_score', 'qa_f1_score', 'rouge_score'}
    selected = [n for n in ast.parse(reference.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {node.name for node in selected} == names
    namespace = dict(re=re, string=string, Counter=Counter, Rouge=Rouge)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(reference), 'exec'), namespace)
    return namespace, reference


def assert_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value), 'Nonfinite serialized numeric result'


def verify_geometry(record, arm, prompt_length, layers, kv_heads, query_heads, head_dim):
    steps = record['generated_tokens'] - 1
    width = record['decode_attended_width']
    assert width['layer_calls'] == layers * steps
    assert width['sum'] == sum(width['per_layer_sum'])
    if not steps:
        assert width['min'] is None and width['max'] is None and width['per_layer_sum'] == []
    else:
        assert len(width['per_layer_sum']) == layers
        assert 0 < width['min'] <= width['max']
        assert layers * steps * width['min'] <= width['sum'] <= layers * steps * width['max']
    initial_len = min(prompt_length, 1024) if arm != 'full' else prompt_length
    if arm == 'full':
        expected_widths = list(range(prompt_length + 1, prompt_length + steps + 1))
        final_len = prompt_length + steps
    elif arm == 'snapkv_faithful':
        expected_widths = list(range(initial_len + 1, initial_len + steps + 1))
        final_len = initial_len + steps
    elif not arm.startswith('tiered_'):
        expected_widths = [min(prompt_length + i, 1024) + 1 for i in range(steps)]
        final_len = min(prompt_length + steps, 1024)
    else:
        expected_widths = None
        final_len = None
        hot = 1024 if arm == 'tiered_hot1024' else 992
        if steps:
            assert width['max'] <= hot + 32 + 1
    if expected_widths:
        assert width['min'] == min(expected_widths) and width['max'] == max(expected_widths)
        assert width['per_layer_sum'] == [sum(expected_widths)] * layers
    for field, expected_len in [('geometry_after_prefill_before_first_decode', initial_len),
                                ('geometry_after_last_model_call_and_policy_commit', final_len)]:
        geom = record[field]
        assert len(geom['layers']) == layers
        assert geom['unique_payload_backing_bytes'] > 0
        storage_sizes = []
        for i, layer in enumerate(geom['layers']):
            assert layer['layer'] == i
            heads = query_heads if arm == 'snapkv_faithful' else kv_heads
            key = layer['visible_k']
            val = layer['visible_v']
            assert key['shape'] == val['shape']
            assert key['shape'][0] == 1 and key['shape'][1] == heads and key['shape'][3] == head_dim
            if not arm.startswith('tiered_'):
                assert key['shape'][2] == expected_len
            else:
                hot = 1024 if arm == 'tiered_hot1024' else 992
                assert 0 < key['shape'][2] <= hot + 32
                tiers = layer['tier_payloads']
                assert tiers['sram_k']['shape'] == tiers['sram_v']['shape']
                assert tiers['stt_k']['shape'] == tiers['stt_v']['shape']
                assert tiers['sram_k']['shape'][1] <= hot
                assert tiers['stt_k']['shape'][1] <= 2048
                assert layer['cold_live_rows'] + layer['cold_shadow_rows'] == tiers['stt_k']['shape'][1]
            tensors = [key, val] + list(layer.get('tier_payloads', {}).values())
            for tensor in tensors:
                assert tensor['dtype'] == 'torch.bfloat16'
                assert tensor['device'].startswith('cuda')
                assert tensor['logical_bytes'] == math.prod(tensor['shape']) * 2
                assert tensor['backing_storage_bytes'] >= tensor['logical_bytes']
                storage_sizes.append(tensor['backing_storage_bytes'])
        assert max(storage_sizes) <= geom['unique_payload_backing_bytes'] <= sum(storage_sizes)
    assert record['cache_stats']['decode_steps'] == steps
    assert all(value >= 0 for value in record['cache_stats'].values()
               if isinstance(value, (int, float)))
    if not arm.startswith('tiered_') and 'modeled_latency_us' in record['cache_stats']:
        from cost_model import CostModel
        heads = query_heads if arm == 'snapkv_faithful' else kv_heads
        latency, energy = CostModel(bytes_per_elem=2).baseline_attention_cost(width['sum'], heads, head_dim)
        for name, value in [('modeled_latency_us', latency), ('modeled_energy_nj', energy)]:
            assert math.isclose(record['cache_stats'][name], value, abs_tol=.051, rel_tol=1e-10), 'Modeled baseline cost disagrees with actual attended widths'


def summarize(rows, arms):
    if not rows:
        return {'count': 0, 'means': {a: None for a in arms}}
    return {'count': len(rows), 'means': {a: float(np.mean([r['scores'][a] for r in rows])) for a in arms}}


def primary_bootstrap(rows, candidate, control, draws=10000, seed=0):
    """Paired document resampling within task, then equal-task macro mean."""
    by_task = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task[row['task']][row['document_group_sha256']].append(
            row['scores'][candidate] - row['scores'][control])
    assert by_task
    rng = np.random.default_rng(seed)
    sampled_task_means = []
    point_task_means = []
    summaries = {}
    for task in sorted(by_task):
        groups = by_task[task]
        keys = sorted(groups)
        sums = np.array([sum(groups[k]) for k in keys], dtype=np.float64)
        sizes = np.array([len(groups[k]) for k in keys], dtype=np.float64)
        indexes = rng.integers(0, len(keys), size=(draws, len(keys)))
        means = sums[indexes].sum(axis=1) / sizes[indexes].sum(axis=1)
        point = float(sums.sum() / sizes.sum())
        sampled_task_means.append(means)
        point_task_means.append(point)
        summaries[task] = {'rows': int(sizes.sum()), 'document_groups': len(keys),
                           'paired_mean_difference': point,
                           'descriptive_unadjusted_ci95': np.percentile(means, [2.5, 97.5]).tolist()}
    macro_draws = np.mean(sampled_task_means, axis=0)
    return {'candidate': candidate, 'control': control,
            'endpoint': 'equal-task macro mean paired score difference, percentage points',
            'point_estimate': float(np.mean(point_task_means)),
            'ci95': np.percentile(macro_draws, [2.5, 97.5]).tolist(),
            'draws': draws, 'seed': seed, 'task_order': sorted(by_task),
            'method': 'resample target-document groups with replacement independently within each task; retain all examples within each selected group; equal weight task means',
            'per_task_descriptive': summaries,
            'interpretation': 'Preregistered quality contrast under disclosed unequal cache geometry; not a SOTA or equal-physical-memory test'}


def self_test():
    rows = [{'task': t, 'document_group_sha256': g, 'scores': {'a': 7., 'b': 2.}}
            for t, groups in [('x', ['1', '1', '2']), ('y', ['3', '4'])] for g in groups]
    result = primary_bootstrap(rows, 'a', 'b', draws=1000)
    assert result['point_estimate'] == 5. and result['ci95'] == [5., 5.]
    assert result['per_task_descriptive']['x']['document_groups'] == 2
    assert summarize(rows, ['a', 'b'])['means'] == {'a': 7., 'b': 2.}
    assert summarize([], ['a'])['count'] == 0
    metrics, _ = reference_metrics()
    assert metrics['qa_f1_score']('the cat', 'cat') == 1.
    assert metrics['rouge_score']('hello world', 'hello world') > .999
    def payload(shape):
        size = math.prod(shape) * 2
        return {'shape': shape, 'dtype': 'torch.bfloat16', 'device': 'cuda:0',
                'logical_bytes': size, 'backing_storage_bytes': size}
    for arm in ['full', 'streamingllm', 'h2o_adapted', 'snapkv_adapted',
                'tiered_hot1024', 'tiered_hot992', 'snapkv_faithful']:
        tiered = arm.startswith('tiered_')
        heads = 4 if arm == 'snapkv_faithful' else 2
        initial = 1100 if arm == 'full' else (1056 if arm == 'tiered_hot1024' else 1024)
        final = initial + 2 if arm in ('full', 'snapkv_faithful') else initial
        attended = [initial + 1, initial + 2] if arm in ('full', 'snapkv_faithful') else [initial + 1] * 2
        def geom(length):
            layer = {'layer': 0, 'visible_k': payload([1, heads, length, 8]),
                     'visible_v': payload([1, heads, length, 8])}
            if tiered:
                hot = 1024 if arm == 'tiered_hot1024' else 992
                layer['tier_payloads'] = {name: payload([2, hot if name.startswith('sram') else 200, 8])
                    for name in ('sram_k', 'sram_v', 'stt_k', 'stt_v')}
                layer['cold_live_rows'] = 100
                layer['cold_shadow_rows'] = 100
            sizes = [layer['visible_k']['backing_storage_bytes'], layer['visible_v']['backing_storage_bytes']]
            sizes += [t['backing_storage_bytes'] for t in layer.get('tier_payloads', {}).values()]
            return {'layers': [layer], 'unique_payload_backing_bytes': sum(sizes)}
        toy = {'generated_tokens': 3, 'decode_attended_width': {
            'layer_calls': 2, 'min': min(attended), 'max': max(attended),
            'sum': sum(attended), 'per_layer_sum': [sum(attended)]},
            'geometry_after_prefill_before_first_decode': geom(initial),
            'geometry_after_last_model_call_and_policy_commit': geom(final),
            'cache_stats': {'decode_steps': 2}}
        verify_geometry(toy, arm, 1100, 1, 2, 4, 8)
        bad = copy.deepcopy(toy)
        bad['decode_attended_width']['sum'] += 1
        try:
            verify_geometry(bad, arm, 1100, 1, 2, 4, 8)
        except AssertionError:
            pass
        else:
            raise AssertionError('Inconsistent attended-width accounting accepted')
    try:
        assert_finite({'bad': [float('nan')]})
    except AssertionError:
        pass
    else:
        raise AssertionError('Nonfinite validator failed')
    print(json.dumps({'synthetic_unit_checks': 'passed', 'geometry_arms': 7,
        'malformed_widths_rejected': 7, 'actual_evaluation_verified': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, default=BASE / 'verified_results.json')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    plan_path = BASE / 'plan.json'
    plan = load(plan_path)
    expected = expected_records(plan)
    complete, progress = completion_gate(plan, expected)
    if args.progress:
        print(json.dumps(progress, indent=2))
        return
    if not complete:
        raise SystemExit(f'Refusing final verification/aggregation: {progress["completed_request_files"]}/10318 request files; complete shards {progress["shards_marked_complete"]}')
    assert not args.output.exists(), 'Preserve previous verification; choose a distinct output path for an independent rerun'
    from provenance import model_artifact_identity, runtime_identity, read_jsonl, atomic_json
    from transformers import AutoTokenizer, AutoConfig
    import longbench_eval as bench
    plan_hash = sha(plan_path)
    freeze = load(BASE / 'freeze.json')
    assert freeze['plan_sha256'] == plan_hash
    assert sha(BASE / 'source.tar.gz') == freeze['source_archive_sha256']
    assert sha(BASE / 'run_full.py') == plan['runner_sha256']
    assert sha(BASE / 'partitions.json') == plan['partition_sha256']
    assert sha(BASE / 'prepare_plan.py') == plan['preparation_sha256']
    assert all(sha(ROOT / p) == digest for p, digest in plan['source_sha256'].items())
    assert runtime_identity() == plan['runtime']
    assert model_artifact_identity(plan['model']) == plan['model_identity']
    partitions = load(BASE / 'partitions.json')
    for key, filekey in [('inventory_sha256', 'inventory_file'), ('original_sha256', 'original_file')]:
        assert sha(ROOT / partitions[filekey]) == partitions[key]
    partition_rows = {(p['task'], p['source_zero_based_index']): p for p in partitions['rows']}
    assert len(partition_rows) == 1474
    for shard in range(2):
        completion = load(BASE / 'records' / f'shard_{shard}' / 'complete.json')
        assert completion['execution_complete'] and completion['plan_sha256'] == plan_hash
        assert completion['total_population_rows'] == 1474
        runmeta = load(BASE / 'records' / f'shard_{shard}' / 'runtime.json')
        assert runmeta['plan_sha256'] == plan_hash and runmeta['runtime'] == plan['runtime']
        assert runmeta['shard'] == shard and runmeta['shards'] == 2
    metrics, reference_path = reference_metrics()
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    config = AutoConfig.from_pretrained(plan['model'], local_files_only=True)
    layers = config.num_hidden_layers
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    rows = []
    record_hashes = {}
    global_index = 0
    for task in plan['tasks']:
        dataset_path = ROOT / plan['dataset_files'][task]
        assert sha(dataset_path) == plan['dataset_sha256'][task]
        data = read_jsonl(dataset_path)
        assert len(data) == plan['counts'][task]
        for index, item in enumerate(data):
            sid = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            assert sid == plan['sample_ids'][task][index]
            partition = partition_rows[(task, index)]
            assert partition['sample_id'] == sid
            target = item['input'].rsplit('\nQuestion:', 1)[0] if task == 'triviaqa' else item['context']
            doc_key = hashlib.sha256((task + '\0' + ' '.join(target.casefold().split())).encode()).hexdigest()
            assert doc_key == partition['document_group_sha256']
            prompt = plan['protocol']['prompt_templates'][task].format(**item)
            if task not in plan['protocol']['no_chat_tasks']:
                prompt = '[INST] ' + prompt + ' [/INST]'
            token_ids = tokenizer(prompt, return_tensors='np').input_ids
            untruncated = token_ids.shape[1]
            if untruncated > 31500:
                token_ids = np.concatenate([token_ids[:, :15750], token_ids[:, -15750:]], axis=1)
            token_hash = hashlib.sha256(token_ids.tobytes()).hexdigest()
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            scores = {}
            first_tokens = set()
            resource = {}
            shift = global_index % 7
            expected_order = plan['arms'][shift:] + plan['arms'][:shift]
            for arm in plan['arms']:
                path = BASE / 'records' / f'shard_{global_index % 2}' / f'{task}_{index:04d}_{arm}.json'
                record_hashes[str(path.relative_to(ROOT))] = sha(path)
                record = load(path)
                assert_finite(record)
                assert record['execution_complete'] and record['plan_sha256'] == plan_hash
                assert record['task'] == task and record['arm'] == arm and record['sample_id'] == sid
                assert record['physical_file_row_index'] == index and record['global_row_index'] == global_index
                assert record['shard'] == global_index % 2 and record['shards'] == 2
                assert record['balanced_arm_order'] == expected_order
                assert record['input_ids_sha256'] == token_hash and record['prompt_sha256'] == prompt_hash
                assert record['untruncated_prompt_tokens'] == untruncated
                assert record['prompt_tokens'] == token_ids.shape[1]
                assert record['dataset_length'] == item.get('length')
                assert record['answers'] == item['answers']
                assert record['requested_backends'] == {'prefill': 'sdpa', 'decode': 'eager'}
                assert record['all_model_logits_finite'] and record['hooks_and_groups_restored']
                generated = record['generated_token_ids']
                assert all(isinstance(v, int) and 0 <= v < config.vocab_size for v in generated)
                assert len(generated) == record['generated_tokens']
                limit = plan['protocol']['max_gen'][task]
                assert 1 <= len(generated) <= limit
                assert tokenizer.eos_token_id not in generated[:-1]
                assert len(generated) == limit or generated[-1] == tokenizer.eos_token_id
                first_tokens.add(generated[0])
                text = tokenizer.decode(generated, skip_special_tokens=True)
                assert text == record['prediction']
                scored = text.lstrip('\n').split('\n')[0] if task in ('triviaqa', 'trec', 'samsum', 'lsht') else text
                assert scored == record['scored_prediction']
                name = plan['protocol']['metrics'][task]
                assert record['metric_function'] == name and name in ('qa_f1_score', 'rouge_score')
                score = 100 * max(metrics[name](scored, answer, all_classes=item.get('all_classes', [])) for answer in item['answers'])
                assert abs(score - record['score']) < 1e-9, f'Raw score mismatch: {path}'
                assert 0 <= score <= 100 + 1e-9
                verify_geometry(record, arm, token_ids.shape[1], layers,
                                config.num_key_value_heads, config.num_attention_heads, head_dim)
                scores[arm] = score
                resource[arm] = {'initial_payload_backing_bytes': record['geometry_after_prefill_before_first_decode']['unique_payload_backing_bytes'],
                    'final_payload_backing_bytes': record['geometry_after_last_model_call_and_policy_commit']['unique_payload_backing_bytes'],
                    'decode_attended_width': record['decode_attended_width'], 'cache_stats': record['cache_stats']}
            assert len(first_tokens) == 1, f'Common uncompressed prefill first-token invariant failed: {task}/{index}'
            rows.append({'task': task, 'source_zero_based_index': index, 'sample_id': sid,
                'document_group_sha256': doc_key, 'partition': partition['partition'],
                'original135': partition['original135'], 'dataset_length': item.get('length'),
                'scores': scores, 'resources': resource})
            global_index += 1
    assert len(rows) == 1474 and len(record_hashes) == 10318
    task_summaries = {}
    for task in plan['tasks']:
        selected = [r for r in rows if r['task'] == task]
        per_partition = {name: summarize([r for r in selected if r['partition'] == name], plan['arms'])
                         for name in ('consumed_or_document_overlap', 'unconsumed_as_of_inventory')}
        strata = {label: summarize([r for r in selected if r['dataset_length'] is not None and predicate(r['dataset_length'])], plan['arms'])
                  for label, predicate in [('0-4k', lambda n: n < 4000), ('4-8k', lambda n: 4000 <= n < 8000), ('8k+', lambda n: n >= 8000)]}
        task_summaries[task] = {'full_population_descriptive': summarize(selected, plan['arms']),
            'original135_reproduction_stratum': summarize([r for r in selected if r['original135']], plan['arms']),
            'recorded_use_partitions': per_partition,
            'source_dataset_length_strata': strata,
            'length_stratum_note': 'Uses original dataset length field, not post-truncation token count; six-file mixture is not the complete standard LongBench benchmark'}
    unconsumed = [r for r in rows if r['partition'] == 'unconsumed_as_of_inventory']
    assert set(r['task'] for r in unconsumed) == set(plan['tasks']) - {'narrativeqa'}
    contrast = plan['analysis']['primary_contrast']
    primary = primary_bootstrap(unconsumed, contrast['candidate'], contrast['control'])
    report = {'verified': True, 'execution_complete': True, 'plan_sha256': plan_hash,
        'verifier_sha256': sha(__file__), 'reference_metrics_sha256': sha(reference_path),
        'verified_request_count': len(record_hashes), 'verified_population_rows': len(rows),
        'checks': ['frozen source/model/data/runtime identities', 'complete expected two-shard record set',
            'independent unchanged upstream metric AST rescoring', 'tokenizer decoding and re-tokenized prompt hashes',
            'sample and target-document identities/partitions', 'greedy EOS/max-gen and common first-token invariants',
            'serialized finite flags and numerical values', 'KV geometry/dtype/capacity and attended-width accounting',
            'logical step count and cleanup flags'],
        'task_summaries': task_summaries,
        'full_population_equal_task_macro_descriptive': {a: float(np.mean([task_summaries[t]['full_population_descriptive']['means'][a] for t in plan['tasks']])) for a in plan['arms']},
        'primary_contrast': primary,
        'resource_scope': plan['analysis']['resource_scope'],
        'limitations': ['Serialized all-logits-finite and hooks-restored flags are verified assertions from runtime; this verifier does not replay model forward calls.',
            'Existing controls labeled adapted remain adaptations; faithful SnapKV has expanded and growing KV resources.',
            'Full population is descriptive and includes consumed examples; recorded-use partitions do not prove absence of unrecorded or training exposure.',
            'Quality run does not verify distinct latency, physical STT-RAM, context-scaling or headroom experiments.'],
        'record_sha256': record_hashes, 'rows': rows}
    atomic_json(args.output, report)
    print(json.dumps({'verified': True, 'requests': 10318, 'output': str(args.output),
                      'primary_contrast': primary}, indent=2))


if __name__ == '__main__':
    main()
