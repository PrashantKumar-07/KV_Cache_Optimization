"""Reject unsafe pool geometry before any decode mutation or native dispatch."""
from types import SimpleNamespace
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from bounded_pinned_pool import BoundedPoolPolicy
from direct_shared_pool import DirectScorePolicy


def prepared():
    policy=BoundedPoolPolicy(sram_budget=991,stt_budget=1090,promotion_budget=32)
    cfg=SimpleNamespace(sram_capacity=991,sttram_capacity=1090)
    policy.layer_caches={i:SimpleNamespace(cfg=cfg,sram_pos=list(range(991)),
        stt_pos=list(range(1090)),capacity=1154) for i in range(2)}
    policy.manager=SimpleNamespace(geometry=(8,1154))
    policy._vis_len={0:1023,1:1023}
    return policy


def past(appends):
    return tuple((torch.empty(1,1,1023+n,1),None) for n in appends)


@pytest.mark.parametrize('fault',('append','layers','ragged','hot','cold','budget','mutated_append','capacity'))
def test_unsafe_geometry_rejected_before_decode(fault,monkeypatch):
    policy=prepared();incoming=past((1,1));called=[]
    monkeypatch.setattr(DirectScorePolicy,'_decode_step',lambda *args:called.append(args))
    if fault=='append':incoming=past((33,33))
    elif fault=='layers':policy.layer_caches.pop(1)
    elif fault=='ragged':incoming=past((1,2))
    elif fault=='hot':policy.layer_caches[0].sram_pos.append(991)
    elif fault=='cold':policy.layer_caches[0].stt_pos.append(1090)
    elif fault=='budget':policy.promotion_budget=33
    elif fault=='mutated_append':policy.max_decode_append=64;incoming=past((64,64))
    else:policy.layer_caches[0].capacity+=1
    before=(policy.decode_step_count,tuple((tuple(c.sram_pos),tuple(c.stt_pos)) for c in policy.layer_caches.values()))
    with pytest.raises(ValueError):policy._decode_step(incoming)
    assert not called
    assert before==(policy.decode_step_count,tuple((tuple(c.sram_pos),tuple(c.stt_pos)) for c in policy.layer_caches.values()))
    # Fake caches above deliberately have no native cleanup methods.
    policy.layer_caches={};policy.manager=None


def test_supported_append_delegates_unchanged(monkeypatch):
    policy=prepared();incoming=past((32,32));marker=object()
    monkeypatch.setattr(DirectScorePolicy,'_decode_step',lambda self,value:marker if value is incoming else None)
    assert policy._decode_step(incoming) is marker
    policy.layer_caches={};policy.manager=None


@pytest.mark.parametrize('options',({'max_decode_append':0},{'max_decode_append':True},
    {'max_decode_append':1.5},{'sram_budget':32,'promotion_budget':32,'max_decode_append':2},
    {'sram_budget':63,'start_size':4,'recent_size':128}))
def test_invalid_constructor_geometry(options):
    with pytest.raises(ValueError):BoundedPoolPolicy(**options)


def test_only_added_constructor_protection_guard_since_model_gates():
    import ast
    import tarfile
    root=Path(__file__).resolve().parents[1]
    name='experiments/bounded_pinned_pool.py'
    with tarfile.open(root/'results/hardware/bounded_pinned_pool_20261002/model_source.tar.gz') as archive:
        previous=ast.parse(archive.extractfile(name).read())
    current=ast.parse((root/name).read_text())
    constructor=next(n for n in ast.walk(current) if isinstance(n,ast.FunctionDef) and n.name=='__init__' and any(
        isinstance(node,ast.arg) and node.arg=='max_decode_append' for node in n.args.kwonlyargs))
    guards=[n for n in constructor.body if isinstance(n,ast.If) and 'Hard protection exceeds' in ast.unparse(n)]
    assert len(guards)==1
    constructor.body.remove(guards[0])
    assert ast.dump(current)==ast.dump(previous)
