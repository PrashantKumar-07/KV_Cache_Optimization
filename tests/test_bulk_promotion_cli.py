"""Prove opt-in composition and reject unsupported flag combinations before I/O."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import evaluate_query_selection as evaluator
from bulk_promotion_policy import BulkCache, BulkPrepare
from batched_eviction_policy import BatchedEvictionPolicy
from fused_bounded_pool import FusedBoundedCache


def test_bulk_measured_policy_preserves_original_coordinator_and_default_cache():
    options=dict(sram_budget=1024,stt_budget=2048,start_size=4,recent_size=128)
    ordinary=evaluator.MeasuredReuseCpuPolicy(**options)
    bulk=evaluator.MeasuredBulkCpuPolicy(**options)
    try:
        assert ordinary.direct_cache_cls.func is FusedBoundedCache
        assert bulk.direct_cache_cls.func is BulkCache
        assert bulk._prepare_decode_reads.__func__ is BatchedEvictionPolicy._prepare_decode_reads
        assert BulkPrepare in type(bulk).__mro__ and BulkPrepare not in type(ordinary).__mro__
        assert ordinary.reserved_promotion_budget==bulk.reserved_promotion_budget
        assert ordinary.max_decode_append==bulk.max_decode_append
    finally:
        ordinary.reset();bulk.reset()


@pytest.mark.parametrize('missing',('reuse','fused','bounded','direct','batched','query'))
def test_invalid_bulk_combinations_fail_before_dataset_or_model_access(monkeypatch,missing,capsys):
    flags={'reuse':'--cpu-reuse-visible','fused':'--cpu-fused-compaction','bounded':'--cpu-bounded-pool',
        'direct':'--cpu-direct-shared','batched':'--cpu-batched-scores','query':'--real-query-prefill'}
    def forbidden(*args,**kwargs):raise AssertionError('Invalid flags reached I/O')
    monkeypatch.setattr(evaluator.bench,'task_data_path',forbidden)
    monkeypatch.setattr(evaluator.AutoModelForCausalLM,'from_pretrained',forbidden)
    monkeypatch.setattr(sys,'argv',['evaluate','--data-dir','unused','--tasks','qasper','--strategy','lru',
        '--output','unused.json','--cpu-bulk-promotion',*[flag for key,flag in flags.items() if key!=missing]])
    with pytest.raises(SystemExit) as error:evaluator.main()
    assert error.value.code==2 and 'requires' in capsys.readouterr().err
