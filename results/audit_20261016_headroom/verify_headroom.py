"""Independent headroom verification. Refuse partial aggregates or figures."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FULL = ROOT / 'results/full_evaluation_20261015'
spec = importlib.util.spec_from_file_location('verified_full_helpers', FULL / 'verify_results.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
sha, load = helpers.sha, helpers.load


def expected_records(plan):
    result = {}
    global_index = 0
    for task in plan['tasks']:
        for index, sid in enumerate(plan['sample_ids'][task]):
            for arm in plan['arms_by_task'][task]:
                path = BASE / 'records' / f'shard_{global_index % 2}' / f'{task}_{index:04d}_{arm}.json'
                result[path] = (task, index, sid, global_index, arm)
            global_index += 1
    assert len(result) == plan['new_predictions'] == 3822
    return result


def check_tiered_cost(record, layers, heads, head_dim):
    """Reconstruct all disclosed analytical event costs from independent totals.

    Exposure totals contain initial and all post-decode visible sets. Remove
    the final set (never consumed by a forward) to get actual cold reads.
    """
    from cost_model import CostModel
    stats = record['cache_stats']
    steps = record['generated_tokens'] - 1
    assert stats['decode_steps'] == steps
    assert stats['stt_exposure_samples'] == steps + 1
    assert stats['demoted'] == stats['paid_writes'] + stats['writes_saved']
    final = record['geometry_after_last_model_call_and_policy_commit']['layers']
    final_visible = {layer['visible_k']['shape'][2] for layer in final}
    assert len(final_visible) == 1
    final_hot = min(layer['tier_payloads']['sram_k']['shape'][1] for layer in final)
    final_cold = next(iter(final_visible)) - final_hot
    assert 0 <= final_cold <= 32
    cold_read_rows = (stats['stt_exposed_total'] - final_cold) * layers
    hot_read_rows = record['decode_attended_width']['sum'] - cold_read_rows
    assert cold_read_rows >= 0 and hot_read_rows >= 0
    elems_per_pair = 2 * heads * head_dim
    cm = CostModel(bytes_per_elem=2, tier_bytes_per_elem={'STT-RAM': 2})
    promote = cm.promote_cost(stats['promoted'] * elems_per_pair)
    demote = cm.demote_cost(stats['paid_writes'] * elems_per_pair)
    drop = cm.drop_cost(stats['dropped'] * elems_per_pair)
    hot = cm.gpu_vram_read_cost(hot_read_rows * elems_per_pair)
    cold = (cm.read_latency_us('STT-RAM', cold_read_rows * elems_per_pair),
            cm.read_energy_nj('STT-RAM', cold_read_rows * elems_per_pair))
    latency = sum(value[0] for value in [promote, demote, drop, hot, cold])
    energy = sum(value[1] for value in [promote, demote, drop, hot, cold])
    for key, value in [('modeled_latency_us', latency), ('modeled_energy_nj', energy)]:
        assert math.isclose(stats[key], value, abs_tol=.051, rel_tol=1e-10), f'Analytical event accounting mismatch: {key}'
    denominator = stats['paid_writes'] + stats['writes_saved']
    savings = 100 * stats['writes_saved'] / denominator if denominator else 0.
    assert abs(stats['write_savings_pct'] - round(savings, 1)) < 1e-9
    return {'cold_layer_token_reads': cold_read_rows, 'hot_layer_token_reads_including_new': hot_read_rows,
            'recomputed_modeled_latency_us': latency, 'recomputed_modeled_energy_nj': energy}


def summarize(records):
    scores = [r['score'] for r in records]
    paid = sum(r['cache_stats']['paid_writes'] for r in records)
    saved = sum(r['cache_stats']['writes_saved'] for r in records)
    return {'count': len(records), 'mean_score': float(np.mean(scores)) if scores else None,
        'paid_writes': paid, 'writes_saved': saved,
        'write_savings_percent': 100 * saved / (paid + saved) if paid + saved else None,
        'modeled_latency_us': sum(r['cache_stats']['modeled_latency_us'] for r in records),
        'modeled_energy_nj': sum(r['cache_stats']['modeled_energy_nj'] for r in records),
        'unit': 'per-layer-token decode demotion events; excludes initial prompt placement, metadata and physical endurance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    plan_path = BASE / 'plan.json'
    plan = load(plan_path)
    expected = expected_records(plan)
    present = set((BASE / 'records').glob('shard_*/*_*.json'))
    assert not present - set(expected), 'Unexpected request files'
    count = Counter((expected[p][0], expected[p][-1]) for p in present)
    complete_shards = [s for s in range(2) if (BASE / 'records' / f'shard_{s}' / 'complete.json').exists()]
    if args.progress:
        print(json.dumps({'new_completed': len(present), 'new_expected': len(expected),
            'reused_reference_count': 1474, 'complete_shards': complete_shards,
            'per_task_fraction_counts': {t: {a: count[(t, a)] for a in plan['arms_by_task'][t]} for t in plan['tasks']},
            'scope': 'file counts only; no scores opened or aggregated'}, indent=2))
        return
    if len(present) != len(expected) or complete_shards != [0, 1]:
        raise SystemExit(f'Refusing partial headroom verification/aggregation: {len(present)}/3822 fresh records; complete shards {complete_shards}')
    output = BASE / 'verified_results.json'
    assert not output.exists(), 'Preserve previous verification'
    from provenance import model_artifact_identity, runtime_identity, read_jsonl, atomic_json
    from transformers import AutoTokenizer, AutoConfig
    plan_hash = sha(plan_path)
    assert load(BASE / 'freeze.json')['plan_sha256'] == plan_hash
    assert sha(BASE / 'run_headroom.py') == plan['runner_sha256']
    assert sha(BASE / 'prepare_plan.py') == plan['preparation_sha256']
    assert sha(FULL / 'run_full.py') == plan['frozen_generator_sha256']
    assert sha(FULL / 'plan.json') == plan['reference_plan_sha256']
    assert sha(FULL / 'verified_results.json') == plan['reference_verification_sha256']
    previous_verification = load(FULL / 'verified_results.json')
    assert previous_verification['verified']
    assert sha(FULL / 'verify_results.py') == previous_verification['verifier_sha256']
    assert sha(ROOT / plan['reference_source_archive']) == plan['reference_source_archive_sha256']
    assert all(sha(ROOT / p) == h for p, h in plan['source_sha256'].items())
    assert model_artifact_identity(plan['model']) == plan['model_identity']
    assert runtime_identity() == plan['runtime']
    for shard in range(2):
        completion = load(BASE / 'records' / f'shard_{shard}' / 'complete.json')
        assert completion['execution_complete'] and completion['plan_sha256'] == plan_hash
        assert completion['total_population_rows'] == 1474
        assert completion['new_request_records'] == sum(meta[3] % 2 == shard for meta in expected.values())
        runtime = load(BASE / 'records' / f'shard_{shard}' / 'runtime.json')
        assert runtime['plan_sha256'] == plan_hash and runtime['runtime'] == plan['runtime']
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    config = AutoConfig.from_pretrained(plan['model'], local_files_only=True)
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    metrics, reference_metrics_path = helpers.reference_metrics()
    summaries = {}
    hashes = {}
    global_index = 0
    for task in plan['tasks']:
        source = ROOT / plan['dataset_files'][task]
        assert sha(source) == plan['dataset_sha256'][task]
        data = read_jsonl(source)
        assert len(data) == plan['counts'][task]
        grouped = {a: [] for a in plan['arms_by_task'][task] + ['frac075']}
        for index, item in enumerate(data):
            sid = helpers.hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            assert sid == plan['sample_ids'][task][index]
            prompt = plan['protocol']['prompt_templates'][task].format(**item)
            if task not in plan['protocol']['no_chat_tasks']:
                prompt = '[INST] ' + prompt + ' [/INST]'
            ids = tokenizer(prompt, return_tensors='np').input_ids
            untruncated = ids.shape[1]
            if untruncated > 31500:
                ids = np.concatenate([ids[:, :15750], ids[:, -15750:]], axis=1)
            token_hash = helpers.hashlib.sha256(ids.tobytes()).hexdigest()
            prompt_hash = helpers.hashlib.sha256(prompt.encode()).hexdigest()
            reference_path = FULL / 'records' / f'shard_{global_index % 2}' / f'{task}_{index:04d}_tiered_hot1024.json'
            reference_relative = str(reference_path.relative_to(ROOT))
            assert sha(reference_path) == plan['reused_reference_records_sha256'][reference_relative]
            records = [(a, BASE / 'records' / f'shard_{global_index % 2}' / f'{task}_{index:04d}_{a}.json') for a in plan['arms_by_task'][task]]
            records += [('frac075', reference_path)]
            first = set()
            for arm, path in records:
                r = load(path)
                hashes[str(path.relative_to(ROOT))] = sha(path)
                helpers.assert_finite(r)
                assert r['execution_complete'] and r['task'] == task and r['sample_id'] == sid
                assert r['physical_file_row_index'] == index and r['global_row_index'] == global_index
                assert r['shard'] == global_index % 2 and r['shards'] == 2
                assert r['plan_sha256'] == (plan['reference_plan_sha256'] if arm == 'frac075' else plan_hash)
                assert r['arm'] == ('tiered_hot1024' if arm == 'frac075' else arm)
                if arm != 'frac075':
                    assert r['fraction'] == plan['fractions'][arm]
                    assert r['reference_fraction075_record'] == reference_relative
                    assert r['reference_fraction075_record_sha256'] == plan['reused_reference_records_sha256'][reference_relative]
                    arms = plan['arms_by_task'][task]
                    shift = global_index % len(arms)
                    assert r['balanced_arm_order'] == arms[shift:] + arms[:shift]
                assert r['input_ids_sha256'] == token_hash and r['prompt_sha256'] == prompt_hash
                assert r['prompt_tokens'] == ids.shape[1] and r['untruncated_prompt_tokens'] == untruncated
                assert r['answers'] == item['answers'] and r['dataset_length'] == item.get('length')
                assert r['requested_backends'] == {'prefill': 'sdpa', 'decode': 'eager'}
                assert r['all_model_logits_finite'] and r['hooks_and_groups_restored']
                generated = r['generated_token_ids']
                assert len(generated) == r['generated_tokens']
                assert all(isinstance(v, int) and 0 <= v < config.vocab_size for v in generated)
                assert 1 <= len(generated) <= plan['protocol']['max_gen'][task]
                assert tokenizer.eos_token_id not in generated[:-1]
                assert len(generated) == plan['protocol']['max_gen'][task] or generated[-1] == tokenizer.eos_token_id
                first.add(generated[0])
                text = tokenizer.decode(generated, skip_special_tokens=True)
                assert text == r['prediction']
                scored = text.lstrip('\n').split('\n')[0] if task == 'triviaqa' else text
                assert scored == r['scored_prediction']
                name = plan['protocol']['metrics'][task]
                assert r['metric_function'] == name
                score = 100 * max(metrics[name](scored, answer) for answer in item['answers'])
                assert abs(score - r['score']) < 1e-9
                helpers.verify_geometry(r, 'tiered_hot1024', ids.shape[1], config.num_hidden_layers,
                                        config.num_key_value_heads, config.num_attention_heads, head_dim)
                check_tiered_cost(r, config.num_hidden_layers, config.num_key_value_heads, head_dim)
                grouped[arm].append(r)
            assert len(first) == 1
            global_index += 1
        summaries[task] = {arm: summarize(records) for arm, records in grouped.items()}
    assert len(hashes) == 5296
    qa_tasks = [t for t in plan['tasks'] if t != 'gov_report']
    figure8 = []
    for arm, fraction in [('frac025', .25), ('frac050', .5), ('frac075', .75), ('frac100', 1.)]:
        values = [summaries[t][arm] for t in qa_tasks]
        savings = [v['write_savings_percent'] for v in values]
        figure8.append({'fraction': fraction, 'mean_qa_f1': float(np.mean([v['mean_score'] for v in values])),
            'equal_task_mean_write_savings_percent': float(np.mean(savings)) if all(v is not None for v in savings) else None,
            'task_count': len(qa_tasks), 'rows': sum(v['count'] for v in values)})
    report = {'verified': True, 'plan_sha256': plan_hash, 'verifier_sha256': sha(__file__),
        'reference_metric_source_sha256': sha(reference_metrics_path), 'fresh_predictions': 3822,
        'reused_fraction075_predictions': 1474, 'independent_analytical_event_cost_checks': 5296,
        'task_fraction_summaries': summaries, 'figure8_numeric_data': figure8,
        'figure15_numeric_data': {t: {a: summaries[t][a] for a in ['frac100', 'frac075']} for t in plan['tasks']},
        'analysis_scope': plan['analysis'], 'raw_record_sha256': hashes,
        'limitation': 'Descriptive fixed mechanism ablation after benchmark exposure; no new operating-point tuning, noninferiority, physical-device or SOTA claim.'}
    atomic_json(output, report)
    print(json.dumps({'verified': True, 'fresh_predictions': 3822,
        'reused_predictions': 1474, 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
