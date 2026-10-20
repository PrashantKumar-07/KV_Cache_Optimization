"""Behavioral regression coverage for the September 2026 review."""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import pytest
import torch
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p) for p in ('src','experiments')]
from tiered_kv_cache import TieredKVCache, TieredConfig
from baselines import FullAttention
import longbench_eval as le
import latency_profile as lp
from provenance import atomic_json, read_jsonl, resume_results
from report_remediation import TASKS, compare
from evaluate_query_selection import QuerySelector
from compare_policy_fastpath import LegacyMetadataPolicy
from test_longbench_policies import _tiny_model, _StubTokenizer, _fake_past


def test_flexicache_observation_preserves_dispatch_and_counts_payload():
    from types import SimpleNamespace as NS
    from run_flexicache_pilot import observe_resources
    calls = []
    sentinel = object()
    def transfer(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    runner = NS(_kv_d2h_tx=NS(kv_cache_d2h_transfer=transfer),
                _kv_h2d_tx=NS(kv_cache_h2d_transfer=transfer))
    output = NS(num_scheduled_tokens={"request": 17})
    pool = NS(num_gpu_blocks=10, num_cpu_blocks=20,
              get_num_free_blocks=lambda: 3, get_num_free_cpu_blocks=lambda: 11)
    manager = NS(block_pools=[pool], pending_free={"request": [[1, 2], [3]]},
                 minmax_block_pools=[NS(num_blocks=9, get_num_free_blocks=lambda: 5)])
    core = NS(scheduler=NS(kv_cache_manager=manager, schedule=lambda: output))
    report = observe_resources(runner, core)
    cache = torch.zeros(2, 5, 16, 128, dtype=torch.bfloat16)
    indices = torch.arange(3)
    for module, name in ((runner._kv_d2h_tx, "kv_cache_d2h_transfer"),
                         (runner._kv_h2d_tx, "kv_cache_h2d_transfer")):
        assert getattr(module, name)(cache, cache, indices, indices, chunk_pairs=2) is sentinel
        assert getattr(module, name)(cache, cache, indices[:0], indices[:0], chunk_pairs=2) is sentinel
    assert len(calls) == 4 and all(k == {"chunk_pairs": 2} for _, k in calls)
    assert all(args[0] is cache and args[1] is cache for args, _ in calls)
    assert report["transfers"] == {direction: {
        "calls": 2, "paired_kv_pages": 3, "payload_bytes": 3 * 8192}
        for direction in ("d2h", "h2d")}
    assert core.scheduler.schedule() is output
    assert report["scheduler_snapshots"] == [{"scheduled_tokens": 17,
        "gpu_pages_in_use": 7, "cpu_pages_in_use": 9, "minmax_pages_in_use": 4,
        "pending_gpu_release_pages": 3}]


@pytest.mark.parametrize("module", ["report_flexicache_resource_gap", "check_flexicache_transfers", "run_flexicache_pilot", "probe_host_hot_scores", "probe_cpu_slots", "probe_batched_cpu_transfers", "check_cpu_batched", "compare_cpu_batched", "profile_cpu_batched", "probe_batched_score_updates", "check_batched_generation", "check_batched_boundaries", "check_snapkv_needles", "check_tiered_query_needles", "check_cpu_query_needles", "audit_unused_documents", "make_fresh_document_holdout", "evaluate_fresh_documents", "report_fresh_documents", "audit_narrativeqa_story_ids", "prepare_narrativeqa_test_stories", "download_narrativeqa_test_stories", "redecode_narrativeqa_stories", "audit_snap_tier_resources", "check_flexicache_prefill", "report_flexicache_pilot_pair"])
def test_flexicache_checks_preserve_existing_evidence(tmp_path, monkeypatch, module):
    import importlib
    output = tmp_path / "frozen.json"
    output.write_text('{"frozen": true}')
    extra = []
    if module == "check_flexicache_transfers":
        extra = ["--build-dir", str(tmp_path)]
    elif module in ("check_batched_generation", "check_batched_boundaries", "check_snapkv_needles", "check_tiered_query_needles", "evaluate_fresh_documents", "download_narrativeqa_test_stories"):
        extra = ["--manifest", str(tmp_path)]
    elif module == "redecode_narrativeqa_stories":
        extra = ["--manifest", str(tmp_path), "--preflight", str(tmp_path), "--labels", str(tmp_path)]
    elif module == "prepare_narrativeqa_test_stories":
        extra = ["--inventory", str(tmp_path), "--metadata", str(tmp_path)]
    elif module == "audit_narrativeqa_story_ids":
        extra = ["--metadata", str(tmp_path)]
    elif module == "report_fresh_documents":
        extra = ["--manifest", str(tmp_path), "--raw", str(tmp_path)]
    elif module == "make_fresh_document_holdout":
        extra = ["--inventory", str(tmp_path)]
    elif module == "check_cpu_query_needles":
        extra = ["--manifest", str(tmp_path), "--reference", str(tmp_path)]
    elif module == "check_flexicache_prefill":
        extra = ["--upstream", str(tmp_path)]
    elif module == "report_flexicache_pilot_pair":
        extra = ["--first", str(tmp_path), "--repeat", str(tmp_path)]
    elif module == "run_flexicache_pilot":
        extra = ["--upstream", str(tmp_path), "--model", str(tmp_path),
                 "--data-file", str(tmp_path), "--expected-token-hash", "unused"]
    monkeypatch.setattr(sys, "argv", [module, "--output", str(output), *extra])
    with pytest.raises(FileExistsError, match="Preserve prior evidence"):
        importlib.import_module(module).main()
    assert output.read_text() == '{"frozen": true}'


@pytest.mark.parametrize("version,code,priorities,valid", [
    (12040, 0, (0, -5), True), (13000, 0, (0, -5), False),
    (12040, 35, (0, 0), False), (12040, 0, (0, 0), False)])
def test_flexicache_cudart_adapter_is_validated_and_scoped(tmp_path, monkeypatch, version, code, priorities, valid):
    from types import SimpleNamespace
    import run_flexicache_pilot as pilot
    library = tmp_path / "libcudart.so.12"
    library.write_bytes(b"runtime fixture")
    def runtime_version(value):
        value._obj.value = version
        return 0
    def priority_range(least, greatest):
        least._obj.value, greatest._obj.value = priorities
        return code
    monkeypatch.setattr(pilot.ctypes, "CDLL", lambda path: SimpleNamespace(
        cudaRuntimeGetVersion=runtime_version, cudaDeviceGetStreamPriorityRange=priority_range))
    original = lambda name: f"original:{name}"
    worker = SimpleNamespace(find_library=original)
    if valid:
        report = pilot.pin_worker_cudart(worker, library)
        assert worker.find_library("cudart") == str(library.resolve())
        assert worker.find_library("other") == "original:other"
        assert report["original_lookup"] == "original:cudart"
        assert report["priority_range"] == list(priorities)
        assert report["query_return_code"] == 0
    else:
        with pytest.raises(ValueError):
            pilot.pin_worker_cudart(worker, library)
        assert worker.find_library is original


def test_flexicache_prefill_adapter_changes_only_pipeline_stages():
    from types import SimpleNamespace
    from flexicache_launch import StageOnePrefill
    calls = []
    compiled = SimpleNamespace(metadata=SimpleNamespace(shared=65536))
    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))
                return compiled
            return launch
    adapter = StageOnePrefill(Kernel())
    grid, argument = object(), object()
    kwargs = dict(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=3, scale=.125)
    assert adapter[grid](argument, **kwargs) is compiled
    assert calls == [(grid, (argument,), kwargs | {"num_stages": 1})]
    assert kwargs["num_stages"] == 3
    assert adapter.calls[0]["compiled_shared_bytes"] == 65536
    with pytest.raises(ValueError, match="Unexpected released"):
        adapter[grid](argument, **(kwargs | {"BLOCK_N": 64}))
    assert len(calls) == 1


def test_simulator_matches_full_without_eviction():
    cfg=TieredConfig(num_heads=2,head_dim=4,sink_size=1,window_size=2,
                     sram_capacity=64,sttram_capacity=8,promote_top_pages=0)
    c=TieredKVCache(cfg);f=FullAttention()
    k,v,q=(torch.randn(2,8,4) for _ in range(3))
    c.initial_bifurcation(k,v,q);f.reset_prompt(k,v,q)
    for i in range(8,14):
        q,k,v=(torch.randn(2,1,4) for _ in range(3))
        torch.testing.assert_close(c.step(q,k,v,i),f.step(q,k,v,i))


def test_empty_candidates_break_hysteresis():
    c=TieredKVCache(TieredConfig(num_heads=1,head_dim=1,sink_size=0,
        window_size=1,sram_capacity=2,sttram_capacity=10))
    k=torch.zeros(1,5,1);c.initial_bifurcation(k,k,k)
    c.stt_was_vram_resident[:]=True
    assert c.resolve_promotions([(0,1)])[0]==[]
    c.resolve_promotions([])
    assert c.resolve_promotions([(0,1)])[0]==[]
    assert c.resolve_promotions([(0,1)])[0]==[(0,1)]


def test_index_hysteresis_matches_singleton_page_path():
    """The real-model fast path preserves singleton-page decisions/state."""
    cfg=TieredConfig(num_heads=1,head_dim=2,sram_capacity=2,
                     sttram_capacity=4,sink_size=0,window_size=0)
    slow=torch.arange(8,dtype=torch.float32).reshape(1,4,2)
    caches=[TieredKVCache(cfg),TieredKVCache(cfg)]
    for cache in caches:
        cache.stt_k=slow.clone();cache.stt_v=slow.clone()
        cache.stt_pos=list(range(4));cache.stt_last_access=torch.zeros(4)
        cache.stt_shadow=torch.tensor([False,True,False,False])
        cache.stt_was_vram_resident=torch.tensor([True,False,False,True])
        cache.stt_peek_streak=torch.tensor([0.,0.,0.,1.])
    pages=[(0,1),(1,2),(2,3),(3,4)]
    promote_pages,deferred_pages=caches[0].resolve_promotions(pages)
    promote_idx,deferred_idx=caches[1].resolve_promotion_indices([0,1,2,3])
    assert promote_idx==[s for s,_ in promote_pages]
    assert deferred_idx==[s for s,_ in deferred_pages]
    assert torch.equal(caches[0].stt_peek_streak,caches[1].stt_peek_streak)


def test_batched_backup_restoration_matches_row_updates():
    base=dict(num_heads=2,head_dim=4,sink_size=1,window_size=2,
              sram_capacity=8,sttram_capacity=64,kv_dtype=torch.bfloat16)
    old=TieredKVCache(TieredConfig(**base,batch_backup_updates=False))
    fast=TieredKVCache(TieredConfig(**base,batch_backup_updates=True))
    prompt=torch.randn(2,30,4,generator=torch.Generator().manual_seed(123))
    for cache in (old,fast):
        cache.initial_bifurcation(prompt,prompt*2,prompt)
        chosen=[i for i,protected in enumerate(cache._hard_protected_mask())
                if not protected][:3]
        cache.sram_cum.fill_(10)
        cache.sram_cum[chosen]=0
        cache.sram_age.fill_(1)
        cache.stt_k=torch.cat((cache.stt_k,cache.sram_k[:,chosen,:]),dim=1)
        cache.stt_v=torch.cat((cache.stt_v,cache.sram_v[:,chosen,:]),dim=1)
        cache.stt_pos += [cache.sram_pos[i] for i in chosen]
        cache.stt_last_access=torch.cat((cache.stt_last_access,torch.zeros(3)))
        cache.stt_shadow=torch.cat((cache.stt_shadow,torch.ones(3,dtype=torch.bool)))
        cache.stt_was_vram_resident=torch.cat(
            (cache.stt_was_vram_resident,torch.ones(3,dtype=torch.bool)))
        cache.stt_peek_streak=torch.cat((cache.stt_peek_streak,torch.ones(3)))
        cache._t=7
    new=torch.ones(2,3,4)
    answers=[cache.evict_and_demote(new,new*2,[30,31,32]) for cache in (old,fast)]
    assert answers[0]==answers[1]
    assert answers[0][2]==3  # three shadow backups restored, no payload writes
    assert old.sram_pos==fast.sram_pos and old.stt_pos==fast.stt_pos
    for name in ('stt_shadow','stt_last_access','stt_peek_streak',
                 'stt_k','stt_v','sram_k','sram_v'):
        assert torch.equal(getattr(old,name),getattr(fast,name)),name


def test_fp8_cold_simulator_sketch_and_step():
    torch.manual_seed(7)
    base=dict(num_heads=2,head_dim=4,sink_size=2,window_size=4,
              sram_capacity=8,sttram_capacity=16,sttram_bifurcation_frac=.75,
              kv_dtype=torch.float32,stt_kv_dtype=torch.float8_e4m3fn)
    cache=TieredKVCache(TieredConfig(**base))
    prompt=torch.randn(2,40,4)
    cache.initial_bifurcation(prompt,prompt,prompt)
    q,k,v=(torch.randn(2,1,4) for _ in range(3))
    out=cache.step(q,k,v,40)
    assert torch.isfinite(out).all()
    assert cache.stt_k.dtype==torch.float8_e4m3fn


def test_legacy_benchmark_wrapper_releases_cache_method_references():
    policy=LegacyMetadataPolicy(sram_budget=16,stt_budget=32,start_size=1,
                                recent_size=4,sttram_bifurcation_frac=.75)
    visible=policy(_fake_past(80))
    next_past=tuple((torch.cat((k,k[:,:,:1,:]),dim=2),
                     torch.cat((v,v[:,:,:1,:]),dim=2)) for k,v in visible)
    policy(next_past)
    assert all('resolve_promotion_indices' not in cache.__dict__
               for cache in policy.layer_caches.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_cache_keeps_hysteresis_on_cpu_without_moving_lru_stamps():
    cfg=TieredConfig(num_heads=2,head_dim=4,sink_size=2,window_size=4,
                     sram_capacity=8,sttram_capacity=16,device='cuda:0')
    c=TieredKVCache(cfg)
    k=torch.randn(2,32,4,device='cuda:0')
    c.initial_bifurcation(k,k,k)
    assert c.stt_last_access.device.type=='cuda'
    assert c.stt_peek_streak.device.type=='cpu'
    c.resolve_promotions([(0,1)])
    c.evict_and_demote(k[:,:1],k[:,:1],32)
    assert len(c.sram_pos)<=cfg.sram_capacity
    assert len(c.stt_pos)<=cfg.sttram_capacity


def test_first_eos_does_not_decode_again():
    model,tok=_tiny_model(),_StubTokenizer()
    original=model.forward;calls=[]
    def forward(*a,**kw):
        out=original(*a,**kw);out.logits.fill_(-100)
        out.logits[:,:,tok.eos_token_id]=100;calls.append(1);return out
    model.forward=forward
    le.generate_with_policy(model,tok,'a b c',5,le.FullCachePolicy(),'cpu')
    assert len(calls)==1


def test_cli_factory_preserves_write_weight():
    args=argparse.Namespace(recent_size=16,budget=64,tiered_vram_budget=64,
        tiered_stt_budget=128,sttram_bifurcation_frac=.75,tiered_stt_expose=8,
        stt_live_floor=1,write_aware_lambda=.05,tiered_kv_dtype='bfloat16',
        tiered_inclusive=False)
    p=le.make_policy('tieredkv',args)
    assert p.write_aware_lambda==.05
    past=((torch.randn(1,2,160,4),torch.randn(1,2,160,4)),)
    p(past)
    assert p.layer_caches[0].cfg.write_aware_lambda==.05
    assert p.layer_caches[0].cfg.inclusive is False


def test_no_shadow_deferred_candidates_survive_promotion_reindex():
    p=le.TieredKVPolicy(sram_budget=16,stt_budget=64,start_size=1,
        recent_size=4,page_size=1,promote_top_pages=2,
        sttram_bifurcation_frac=.75,inclusive=False)
    visible=p(_fake_past(80))
    for layer,cache in p.layer_caches.items():
        last=len(cache.stt_pos)-1
        cache.stt_was_vram_resident[last]=True
        p._stt_attn[layer]={cache.stt_pos[0]:2.0,cache.stt_pos[last]:1.0}
    new_past=tuple((torch.cat((k,torch.zeros_like(k[:,:,:1,:])),dim=2),
                    torch.cat((v,torch.zeros_like(v[:,:,:1,:])),dim=2))
                   for k,v in visible)
    p(new_past)
    assert all(cache.cfg.inclusive is False for cache in p.layer_caches.values())


def test_query_selector_preserves_shared_cache_length():
    model=_tiny_model()
    policy=le.TieredKVPolicy(sram_budget=16,stt_budget=32,start_size=1,
        recent_size=4,sttram_bifurcation_frac=.75)
    selector=QuerySelector(model,policy)
    try:
        le.generate_with_policy(model,_StubTokenizer(),
            ' '.join('token'+str(i) for i in range(70)),8,policy,'cpu')
    finally:
        selector.close()
    assert policy.stats()['decode_steps']>0


def test_latency_includes_policy_time():
    class Slow(le.FullCachePolicy):
        def __call__(self,past):
            time.sleep(.025)
            return past
    stats=lp.profile_one(_tiny_model(),_StubTokenizer(),Slow(),8,
        n_decode=2,n_warmup=0,n_repeat=1,device='cpu')
    assert stats['itl_ms']>=25
    assert stats['ttft_ms']>=25
    assert stats['timing_scope']=='end_to_end_model_policy_token'


def test_latency_budget_tracks_argument():
    p=lp.make_policy('tieredkv',256)
    assert (p.sram_budget,p.stt_budget)==(256,512)
    assert p.sttram_bifurcation_frac==.75


def test_resume_rejects_changed_config(tmp_path):
    path=tmp_path/'run.json';config={'budget':64}
    atomic_json(path,{'config':config,'results':{'full':{}}})
    assert resume_results(path,config)=={'full':{}}
    with pytest.raises(ValueError,match='mismatch'):
        resume_results(path,{'budget':128})


def test_jsonl_preserves_unicode_line_separator_inside_string(tmp_path):
    path = tmp_path / 'records.jsonl'
    path.write_text(json.dumps({'context': 'one\x85two'}, ensure_ascii=False) + '\n'
                    + json.dumps({'context': 'three'}) + '\n')
    assert [row['context'] for row in read_jsonl(path)] == ['one\x85two', 'three']


def test_authored_python_parses():
    root=Path(__file__).resolve().parents[1]
    for folder in ('src','experiments','tests'):
        for p in (root/folder).glob('*.py'):
            ast.parse(p.read_text(),filename=str(p))


def test_cost_width_matches_residency():
    for dtype,width in [(torch.float32,4),(torch.bfloat16,2)]:
        c=TieredKVCache(TieredConfig(kv_dtype=dtype))
        assert c.cm.bytes_per_elem==width


def test_float8_cold_tier_is_stored_compressed_and_exposed_as_bfloat16():
    if not hasattr(torch, 'float8_e4m3fn'):
        pytest.skip('PyTorch has no float8 storage dtype')
    cfg=TieredConfig(num_heads=2,head_dim=8,sink_size=1,window_size=2,
        sram_capacity=4,sttram_capacity=8,kv_dtype=torch.bfloat16,
        stt_kv_dtype=torch.float8_e4m3fn,page_size=2)
    c=TieredKVCache(cfg)
    k=torch.randn(2,16,8,dtype=torch.bfloat16)
    v=torch.randn_like(k); q=torch.randn_like(k)
    c.initial_bifurcation(k,v,q)
    assert c.sram_k.dtype==torch.bfloat16
    assert c.stt_k.dtype==torch.float8_e4m3fn
    assert c.stt_k.element_size()==1
    pk,pv,_=c.get_peek_kv([(0,2)])
    assert pk.dtype==pv.dtype==torch.bfloat16
    assert c.promote([(0,2)])>0
    assert c.sram_k.dtype==c.sram_v.dtype==torch.bfloat16
    before=c.stt_k.shape[1]
    c.stt_k=c.append_stt_storage(c.stt_k,c.sram_k[:,:1])
    c.stt_v=c.append_stt_storage(c.stt_v,c.sram_v[:,:1])
    assert c.stt_k.dtype==torch.float8_e4m3fn and c.stt_k.shape[1]==before+1
    kept=c.index_stt_storage(c.stt_k,[0,before])
    assert kept.dtype==torch.float8_e4m3fn and kept.shape[1]==2


def test_tiered_prefill_chunk_noop_and_bounded_path_run():
    model,tok=_tiny_model(),_StubTokenizer()
    text='one two three four five six seven eight nine ten'
    def run(chunk):
        policy=le.TieredKVPolicy(sram_budget=8,stt_budget=8,start_size=1,
            recent_size=2,kv_dtype=torch.float32,prefill_chunk_size=chunk)
        torch.manual_seed(0)
        return le.generate_with_policy(model,tok,text,2,policy,'cpu',max_ctx=32)
    assert run(0)==run(32)
    assert isinstance(run(4),str)


def test_paired_report_uses_matching_sample_ids():
    results = {}
    for task in TASKS:
        results[task] = {
            'tieredkv': {'score': 2.0, 'n': 2, 'samples': [
                {'sample_id': 'a', 'score': 1.0}, {'sample_id': 'b', 'score': 3.0}]},
            'snapkv': {'score': 1.0, 'n': 2, 'samples': [
                {'sample_id': 'a', 'score': 0.0}, {'sample_id': 'b', 'score': 2.0}]},
        }
    report = compare(results, 'snapkv', bootstrap=100)
    assert report['task_balanced_mean_difference'] == 1.0
    assert report['paired_bootstrap_95pct_interval'] == [1.0, 1.0]
    results[TASKS[0]]['snapkv']['samples'][1]['sample_id'] = 'other'
    with pytest.raises(ValueError, match='Unpaired'):
        compare(results, 'snapkv', bootstrap=100)


def test_ppl_aggregates_all_documents_and_targets():
    from ppl_eval import evaluate_ppl
    model,tok=_tiny_model(),_StubTokenizer()
    a='one two three four five'; b='other words three four six'
    def run(texts):
        return evaluate_ppl(model,tok,texts,le.FullCachePolicy(),max_len=4,checkpoints=[2,4])
    x,y,z=run([a]),run([b]),run([a,b])
    for key in ['2','4']:
        assert z[key]['documents']==2
        assert z[key]['tokens']==2*int(key)
        assert z[key]['nll']==pytest.approx(x[key]['nll']+y[key]['nll'])
    ids=tok(a).input_ids
    with torch.inference_mode():
        logits=model(ids,use_cache=False).logits
    expected=torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,128),
        ids[:,1:].reshape(-1),reduction='sum').item()
    assert x['4']['nll']==pytest.approx(expected,rel=1e-5)


def test_gpu_control_compares_state_without_implying_equal_host_transfers():
    from compare_cpu_batched import check
    reference = dict(state_sha256='state', history_sha256='history',
                     every_visible_state_sha256=['prefill', 'decode'],
                     explicit_h2d_payload_bytes=0, explicit_d2h_payload_bytes=0,
                     initial_d2h_payload_bytes=0, occupied_host_bytes=0,
                     trials=[dict(predicted_token_ids=[1, 2], paid_writes=3,
                                  writes_saved=4, demoted=5, promoted=6, decode_steps=2)])
    candidate = reference | dict(explicit_h2d_payload_bytes=100,
                                explicit_d2h_payload_bytes=200,
                                initial_d2h_payload_bytes=150, occupied_host_bytes=200)
    check(reference, candidate, compare_host_payload=False)
    with pytest.raises(AssertionError, match='explicit_h2d_payload_bytes'):
        check(reference, candidate)
    with pytest.raises(AssertionError, match='state_sha256'):
        check(reference, candidate | dict(state_sha256='changed'), compare_host_payload=False)
    with pytest.raises(AssertionError, match='predicted_token_ids'):
        check(reference, candidate | dict(trials=[reference['trials'][0] | dict(predicted_token_ids=[1, 3])]), compare_host_payload=False)


@pytest.mark.parametrize('regime', ['normal', 'zero', 'mapping', 'unordered', 'heterogeneous'])
def test_batched_score_updates_preserve_scores_and_cold_metadata(regime):
    import torch
    from batched_score_policy import batched_update
    from longbench_eval import TieredKVPolicy
    from probe_batched_score_updates import fixture, assert_equal
    generator = torch.Generator().manual_seed(0)
    cold = 0 if regime == 'zero' else 2
    weights = [torch.rand((1, 2, 1, 4 + cold + 1), generator=generator) for _ in range(4)]
    cumulative = torch.rand(4, 4, generator=generator)
    if regime == 'heterogeneous':
        weights[0] = torch.rand((1, 2, 1, 8), generator=generator)
    left, right = (fixture(weights, cumulative, cold, changed=regime == 'mapping') for _ in range(2))
    if regime == 'unordered':
        for policy in (left, right):
            policy._vis_sram_keep[1] = [3, 1, 2, 0]
    for _ in range(3):
        TieredKVPolicy.update_scores(left, [(None, None)] * 4, weights)
        batched_update(right, [(None, None)] * 4, weights)
        assert_equal(left, right)


def test_snapkv_index_observer_preserves_actual_selection():
    from check_snapkv_needles import observe_indices
    from snapkv_adapter import upstream_module
    module = upstream_module()
    original = module.SnapKVCluster(4, 16, 5, 'avgpool')
    observed = module.SnapKVCluster(4, 16, 5, 'avgpool')
    generator = torch.Generator().manual_seed(41)
    k, q, v = [torch.randn(1, 2, 64, 8, generator=generator, dtype=torch.bfloat16)
               for _ in range(3)]
    seen = []
    observe_indices(observed, lambda indices, length: seen.append((indices.clone(), length)))
    expected = original.update_kv(k, q, v, None, 1)
    actual = observed.update_kv(k, q, v, None, 1)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert len(seen) == 1 and seen[0][1] == 64
    indices = seen[0][0].unsqueeze(-1).expand(-1, -1, -1, 8)
    for source, result in zip((k, v), actual):
        assert torch.equal(result[:, :, :12], source[:, :, :-4].gather(2, indices))
        assert torch.equal(result[:, :, 12:], source[:, :, -4:])
    seen.clear()
    short = [x[:, :, :8] for x in (k, q, v)]
    assert all(torch.equal(a, b) for a, b in zip(
        observed.update_kv(*short, None, 1), original.update_kv(*short, None, 1)))
    assert not seen


def test_resource_accounting_counts_storage_once_and_includes_view_capacity():
    from audit_snap_tier_resources import distinct_storage_bytes
    pool = torch.empty(2, 100, 8, dtype=torch.bfloat16)
    first, second = pool[:, :10], pool[:, 10:20]
    copied = first.clone()
    empty = torch.empty(0)
    assert distinct_storage_bytes([first, second, pool, copied, copied, empty]) == {
        'cpu': pool.untyped_storage().nbytes() + copied.untyped_storage().nbytes()}
    assert distinct_storage_bytes([]) == {}


def test_query_needle_observer_records_consumed_queries_and_fixed_initial_storage():
    from check_batched_boundaries import GpuObserved, FixedTokenizer
    from evaluate_query_selection import PromptQueryCapture
    model = _tiny_model()
    ids = list(range(3, 35))
    geometries = []
    for actual in (False, True):
        policy = GpuObserved(sram_budget=8, stt_budget=16, start_size=1, recent_size=4,
                             stt_expose_quota=2, sttram_bifurcation_frac=.75)
        policy.answer_span = [5, 8]
        hook_counts = [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
        capture = PromptQueryCapture(model, policy) if actual else None
        try:
            le.generate_with_policy(model, FixedTokenizer(_StubTokenizer(), ids), '', 2,
                                    policy, 'cpu', max_ctx=100)
            assert not policy.prompt_q_by_layer
            if actual:
                assert policy.prefill_query_shapes == {0: [4, 4, 4], 1: [4, 4, 4]}
            else:
                assert not policy.prefill_query_shapes
            assert len(policy.initial_cache_rows) == 2
            geometries.append(policy.initial_cache_rows)
        finally:
            if capture is not None:
                capture.close()
            policy.reset()
        assert [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers] == hook_counts
    assert geometries[0] == geometries[1]


def test_story_text_preflight_preserves_dialogue_and_rejects_wrong_markers():
    from download_narrativeqa_test_stories import visible_text,marker_positions
    text,encoding=visible_text(b'<html><style>hidden</style><script>hidden</script><p>Happy <b>Harry</b> Hardon</p><p>by Martin Eaves</p></html>',None,'text/html')
    assert encoding=='utf-8' and 'hidden' not in text
    assert 'Happy Harry Hardon' in text and '\n' in text
    story={'story_start':'Happy Harry Hardon','story_end':'by Martin Eaves'}
    first,last,words,ordered=marker_positions(text,story)
    assert ordered and first==0 and last>first
    assert not marker_positions('by Martin Eaves Happy Harry Hardon',story)[3]
    assert not marker_positions('error page',story)[3]
    assert not marker_positions('Unhappy Harry Hardon by Martin Eaves',story)[3]
    assert visible_text(b'A \x93quoted\x94 line',None,'text/plain')==('A “quoted” line','cp1252')
