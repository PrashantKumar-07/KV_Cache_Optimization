"""Semantic regressions discovered by the October end-to-end audit."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ('src', 'experiments')]
from audit_unused_documents import document_group_key, document_key
from longbench_metrics import DATASET_TO_PROMPT
from longbench_eval import TieredKVPolicy, generate_with_policy, evaluate_task, FullCachePolicy
from tiered_kv_cache import TieredConfig, TieredKVCache
from test_longbench_policies import _tiny_model


@pytest.mark.parametrize('prompt', [0, 1, 2, 20])
def test_config_rejects_eventual_pinned_overflow(prompt):
    cache = TieredKVCache(TieredConfig(num_heads=1, head_dim=2,
        sram_capacity=4, sttram_capacity=4, sink_size=2, window_size=4))
    values = torch.zeros(1, prompt, 2)
    with pytest.raises(ValueError, match='exceeds'):
        cache.initial_bifurcation(values, values, values)


@pytest.mark.parametrize('fraction', [-.5, 1.5, float('nan'), float('inf')])
def test_config_rejects_invalid_cold_fraction(fraction):
    cache = TieredKVCache(TieredConfig(num_heads=1, head_dim=2,
        sram_capacity=4, sttram_capacity=4, sink_size=1, window_size=1,
        sttram_bifurcation_frac=fraction))
    values = torch.zeros(1, 20, 2)
    with pytest.raises(ValueError, match='finite'):
        cache.initial_bifurcation(values, values, values)


def test_prompts_match_reference_and_include_target_dialogue():
    reference = json.loads((ROOT / 'sota/snapkv/experiments/LongBench/config/dataset2prompt.json').read_text())
    assert all(prompt == reference[task] for task, prompt in DATASET_TO_PROMPT.items())
    item = {'context': 'Demonstration dialogue and answer', 'input': 'Distinct target dialogue'}
    assert item['input'] in DATASET_TO_PROMPT['samsum'].format(**item)


def test_target_identity_ignores_demonstrations_and_question_variations():
    a = {'context': 'example A', 'input': 'Target passage\nQuestion: Who?'}
    b = {'context': 'example B', 'input': 'Target passage\nQuestion: When?'}
    assert document_key(a) != document_key(b)  # historical identity is preserved
    assert document_group_key(a, 'triviaqa') == document_group_key(b, 'triviaqa')
    b['input'] = 'Different passage\nQuestion: Who?'
    assert document_group_key(a, 'triviaqa') != document_group_key(b, 'triviaqa')


def test_nonconsecutive_promotion_hits_do_not_promote():
    import numpy as np
    from native_controller import kernel
    policy = TieredKVPolicy(sram_budget=4, stt_budget=4, start_size=1,
        recent_size=1, promotion_budget=1, stt_expose_quota=2)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
    policy(((values, values),))
    cache = policy.layer_caches[0]
    cache.stt_was_vram_resident.fill_(True)
    position = cache.stt_pos[0]
    for hit in (True, False, True):
        policy._stt_attn = {0: {position: 1.0}} if hit else {}
        assert policy._promotion_plan(0, cache) == (([], 1) if hit else (None, 0))
        assert cache.stt_peek_streak[0].item() == (1 if hit else 0)
    frame = dict(shadow=np.zeros(2, np.bool_), resident=np.ones(2, np.bool_),
        streak=np.zeros(2, np.float32), attention=[], inclusive=True, floor=0)
    for hit in (True, False, True):
        frame['attention'] = [(1., 0)] if hit else []
        result = kernel().promotion([frame], 1, 2)[0]
        assert result['hits'] == ([] if hit else None)
        assert result['streak'][0] == (1 if hit else 0)
        frame['streak'] = result['streak']


class FixedTokenizer:
    eos_token_id = -1
    def __call__(self, text, **options):
        return SimpleNamespace(input_ids=torch.arange(3, 23).unsqueeze(0))
    def decode(self, ids, **options):
        return ' '.join(map(str, ids))


@pytest.mark.parametrize('hot,cold', [(4, 4), (32, 32)])
def test_bounded_prefill_preserves_absolute_identity_and_payload(hot, cold):
    model = _tiny_model()
    policy = TieredKVPolicy(sram_budget=hot, stt_budget=cold,
        start_size=1, recent_size=2, prefill_chunk_size=4)
    original = policy._bifurcate
    def bifurcate(past, **options):
        source = [(k.clone(), v.clone()) for k, v in past]
        result = original(past, **options)
        for layer, cache in policy.layer_caches.items():
            positions = options['token_positions'][layer]
            for rows, keys, values in [(cache.sram_pos, cache.sram_k, cache.sram_v),
                                      (cache.stt_pos, cache.stt_k, cache.stt_v)]:
                indices = [positions.index(pos) for pos in rows]
                torch.testing.assert_close(keys, source[layer][0][0, :, indices])
                torch.testing.assert_close(values, source[layer][1][0, :, indices])
            assert cache.metrics.prompt_len == options['prompt_length']
            assert 0 in cache.sram_pos
            assert set(range(max(0, cache.metrics.prompt_len-2), cache.metrics.prompt_len)) <= set(cache.sram_pos)
        return result
    policy._bifurcate = bifurcate
    generate_with_policy(model, FixedTokenizer(), 'prompt', 3, policy, 'cpu')
    for cache in policy.layer_caches.values():
        assert cache.metrics.prompt_len == 20
        assert {20, 21} <= set(cache.sram_pos)
        assert len(cache.sram_pos) <= hot and len(cache.stt_pos) <= cold


def test_bounded_no_drop_matches_full_prefill():
    model = _tiny_model()
    def run(chunk):
        policy = TieredKVPolicy(sram_budget=32, stt_budget=32,
            start_size=1, recent_size=2, prefill_chunk_size=chunk)
        logits = []
        handle = model.register_forward_hook(lambda m, args, out: logits.append(out.logits[:, -1].clone()))
        try:
            generated = generate_with_policy(model, FixedTokenizer(), 'prompt', 3, policy, 'cpu')
        finally:
            handle.remove()
        return generated, logits[-3:]
    full, a = run(0)
    chunked, b = run(4)
    assert full == chunked
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_evaluation_uses_frozen_rows_without_resolving_again(monkeypatch):
    import longbench_eval as bench
    def forbidden(*args, **kwargs):
        raise AssertionError('Dataset must not be independently resolved per arm')
    monkeypatch.setattr(bench, 'load_task_data', forbidden)
    monkeypatch.setattr(bench, 'generate_with_policy', lambda *a, **k: 'answer')
    policy = FullCachePolicy()
    policy.last_generation = {'prompt_tokens': 2, 'generated_tokens': 1}
    score, values = evaluate_task(None, None, 'qasper', policy, 'cpu',
        dataset_rows=[{'context': 'document', 'input': 'question', 'answers': ['answer']}])
    assert score == 100 and values == [100]


@pytest.mark.parametrize('storage_dtype', [torch.float32, torch.bfloat16])
def test_bounded_prefill_casts_storage_to_model_dtype(storage_dtype):
    model = _tiny_model().to(torch.bfloat16)
    policy = TieredKVPolicy(sram_budget=4, stt_budget=4, start_size=1,
        recent_size=2, prefill_chunk_size=4, kv_dtype=storage_dtype)
    generated = generate_with_policy(model, FixedTokenizer(), 'prompt', 2, policy, 'cpu')
    assert len(generated.split()) == 2
    assert all(cache.sram_k.dtype == storage_dtype for cache in policy.layer_caches.values())


@pytest.mark.parametrize('queries', [1, 2, 8])
def test_h2o_multitoken_attention_scores(queries):
    from longbench_eval import H2OPolicy
    policy = H2OPolicy(budget=8, start_size=1, recent_size=2)
    keys = torch.randn(1, 2, 6, 4)
    weights = torch.rand(1, 4, queries, 6).softmax(-1)
    policy.update_scores(((keys, keys),), [weights])
    torch.testing.assert_close(policy.cum_scores[0], weights.float().mean(dim=(0, 1, 2)))


def test_model_identity_detects_tokenizer_and_custom_code_changes(tmp_path):
    from provenance import model_artifact_identity
    (tmp_path / 'config.json').write_text('{}')
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('first')
    code = tmp_path / 'modeling_custom.py'
    code.write_text('VALUE = 1')
    before = model_artifact_identity(tmp_path)
    vocab.write_text('second')
    assert model_artifact_identity(tmp_path) != before
    before = model_artifact_identity(tmp_path)
    code.write_text('VALUE = 2')
    assert model_artifact_identity(tmp_path) != before


def test_real_query_capture_rejects_bounded_prefill_before_installing_hooks():
    from evaluate_query_selection import PromptQueryCapture
    model = _tiny_model()
    before = [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
    policy = TieredKVPolicy(prefill_chunk_size=4)
    with pytest.raises(ValueError, match='requires full prefill'):
        PromptQueryCapture(model, policy)
    assert before == [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]


def test_h2o_preserves_real_scores_on_first_incremental_overflow():
    from longbench_eval import H2OPolicy
    policy = H2OPolicy(budget=4, start_size=1, recent_size=1)
    for count in (4, 5):
        keys = torch.zeros(1, 1, count, 2)
        values = torch.arange(count).view(1, 1, count, 1).expand_as(keys).float()
        weights = torch.zeros(1, 1, 1, count)
        weights[..., 1] = 1
        policy.update_scores(((keys, values),), [weights])
        selected = policy(((keys, values),))
    assert 1 in selected[0][1][0, 0, :, 0].tolist()


def test_no_drop_identical_cache_has_equal_analytical_attention_charge():
    from longbench_eval import FullCachePolicy
    model = _tiny_model()
    full = FullCachePolicy()
    tiered = TieredKVPolicy(sram_budget=32, stt_budget=32, start_size=1,
        recent_size=2, kv_dtype=torch.bfloat16)
    generate_with_policy(model, FixedTokenizer(), 'prompt', 3, full, 'cpu')
    generate_with_policy(model, FixedTokenizer(), 'prompt', 3, tiered, 'cpu')
    latency = sum(cache.metrics.total_latency_us for cache in tiered.layer_caches.values())
    energy = sum(cache.metrics.total_energy_nj for cache in tiered.layer_caches.values())
    assert latency == pytest.approx(full._modeled_latency_us)
    assert energy == pytest.approx(full._modeled_energy_nj)


def test_memory_summary_uses_storage_width_and_empty_sketch_has_zero_cost():
    cache = TieredKVCache(TieredConfig(num_heads=1, head_dim=2, sink_size=0,
        window_size=1, sram_capacity=8, sttram_capacity=8, kv_dtype=torch.float32))
    values = torch.ones(1, 2, 2)
    cache.initial_bifurcation(values, values, values)
    cache.step(values[:, :1], values[:, :1], values[:, :1], 2)
    summary = cache.metrics.summary()
    assert summary['peak_sram_KB'] == round(cache.metrics.peak_sram_tokens * 2 * 2 * 4 / 1024, 2)
    assert cache.metrics.steps[-1].sketch_macs == 0
    assert cache.metrics.steps[-1].lat_sketch_us == 0


def test_heterogeneous_storage_costs_charge_each_tier_byte_width():
    from cost_model import CostModel
    cm = CostModel(bytes_per_elem=2, tier_bytes_per_elem={'STT-RAM': 1})
    assert cm.read_latency_us('STT-RAM', 100) == pytest.approx(100 / 1e9 / 50 * 1e6)
    assert cm.read_energy_nj('STT-RAM', 100) == pytest.approx(100 * 8 * 2 / 1000)
    assert cm.read_energy_nj('GPU_VRAM', 100) == pytest.approx(100 * 2 * 8 * 12 / 1000)


@pytest.mark.parametrize('last_logits', [False, True])
def test_faithful_snap_adapter_generation_accepts_evaluator_options(last_logits):
    from evaluate_snapkv_adapter import generate
    model = _tiny_model()
    policy = SimpleNamespace(capacity=8, window=2)
    groups = [layer.self_attn.num_key_value_groups for layer in model.model.layers]
    with __import__('longbench_eval')._attn_impl(model, 'sdpa'):
        result = generate(model, FixedTokenizer(), 'prompt', 3, policy, 'cpu',
                          prefill_last_logits=last_logits)
    assert len(result.split()) == 3
    assert len(policy.last_prompt_identity['input_ids_sha256']) == 64
    assert groups == [layer.self_attn.num_key_value_groups for layer in model.model.layers]


def test_snap_incremental_scores_follow_compressed_kv_identity():
    from longbench_eval import SnapKVPolicy
    policy = SnapKVPolicy(budget=4, start_size=1, recent_size=1, pool_kernel=1)
    keys = torch.randn(1, 1, 7, 2, generator=torch.Generator().manual_seed(21))
    values = torch.arange(7).reshape(1, 1, 7, 1).expand_as(keys).float()
    weights = torch.arange(1, 8).reshape(1, 1, 1, 7).float()/28
    policy.update_scores(((keys, values),), [weights])
    before = policy._h2o.cum_scores[0].clone()
    selected = policy(((keys, values),))
    positions = selected[0][1][0, 0, :, 0].long()
    torch.testing.assert_close(policy._h2o.cum_scores[0], before[positions])


def test_first_large_real_observation_preserves_prior_and_new_attention():
    from longbench_eval import H2OPolicy
    policy = H2OPolicy(budget=4, start_size=1, recent_size=1)
    control = H2OPolicy(budget=4, start_size=1, recent_size=1)
    keys = torch.randn(1, 1, 5, 2, generator=torch.Generator().manual_seed(4))
    control._initialize_scores(keys, keys, 0)
    weights = torch.zeros(1, 1, 1, 5)
    weights[..., 1] = 1
    policy.update_scores(((keys, keys),), [weights])
    torch.testing.assert_close(policy.cum_scores[0], control.cum_scores[0] + weights[0, 0, 0])


def test_failed_initialization_cannot_silently_bypass_policy_on_retry():
    policy = TieredKVPolicy(sram_budget=2, stt_budget=4, start_size=1, recent_size=2)
    keys = torch.zeros(1,1,8,2)
    with pytest.raises(ValueError, match='exceeds'):
        policy(((keys,keys),))
    with pytest.raises(RuntimeError, match='Reset'):
        policy(((keys,keys),))
    policy.sram_budget = 4
    policy.reset()
    assert policy(((keys,keys),))[0][0].shape[2] <= 4 + policy.stt_expose_quota


@pytest.mark.parametrize('cold_dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_sketch_metadata_cost_uses_bound_dtype_not_cold_payload(cold_dtype):
    from cost_model import DEFAULT_TIERS
    cfg = TieredConfig(num_heads=2, head_dim=4, sram_capacity=4,
        sttram_capacity=32, sink_size=1, window_size=2, page_size=16,
        kv_dtype=torch.bfloat16, stt_kv_dtype=cold_dtype,
        sttram_bifurcation_frac=1.0)
    cache = TieredKVCache(cfg)
    keys = torch.linspace(-1, 1, 2*36*4).reshape(2, 36, 4)
    cache.initial_bifurcation(keys, keys, keys)
    pages = cache._page_bounds()
    bound_bytes = sum(lo.numel()*lo.element_size() + hi.numel()*hi.element_size()
                      for _, _, lo, hi in pages)
    assert bound_bytes == 64
    query = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    cache.step(query, torch.zeros_like(query), torch.zeros_like(query), 36)
    record = cache.metrics.steps[-1]
    expected_us = bound_bytes / 1e9 / DEFAULT_TIERS['STT-RAM'].read_bw_gbps * 1e6
    assert record.lat_sketch_us == pytest.approx(expected_us)


def test_query_evaluation_requires_model_artifact_identity(monkeypatch, tmp_path):
    import evaluate_query_selection as runner
    calls = []
    def capture(location):
        calls.append(location)
        raise RuntimeError('identity captured before model loading')
    monkeypatch.setattr(runner, 'model_artifact_identity', capture)
    monkeypatch.setattr(sys, 'argv', ['query_eval', '--model', str(tmp_path),
        '--data-dir', str(tmp_path), '--tasks', 'qasper', '--strategy', 'lru',
        '--output', str(tmp_path/'out.json')])
    with pytest.raises(RuntimeError, match='identity captured'):
        runner.main()
    assert calls == [str(tmp_path)]
