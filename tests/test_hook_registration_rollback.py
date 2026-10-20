"""Partial hook acquisition must leave unrelated model hooks intact."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from evaluate_query_selection import PromptQueryCapture, QuerySelector
from reserved_decode_append import AppendLease
from snapkv_adapter import SnapKVAdapter

class Policy:
    prefill_chunk_size=0

def fail(*args,**kwargs):
    raise RuntimeError('injected registration failure')

@pytest.mark.parametrize('kind',[PromptQueryCapture,QuerySelector])
def test_layer_hook_failure_rolls_back_only_acquired_hooks(kind,monkeypatch):
    attention=[torch.nn.Linear(2,2) for _ in range(2)]
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=a) for a in attention]))
    existing=attention[0].register_forward_pre_hook(lambda *args:None)
    before=[dict(a._forward_pre_hooks) for a in attention]
    monkeypatch.setattr(attention[1],'register_forward_pre_hook',fail)
    with pytest.raises(RuntimeError,match='injected registration failure'):
        kind(model,Policy())
    assert [dict(a._forward_pre_hooks) for a in attention]==before
    existing.remove()

def test_append_hook_failure_rolls_back_only_acquired_hooks(monkeypatch):
    model=torch.nn.Linear(2,2)
    existing=[model.register_forward_pre_hook(lambda *args:None),
              model.register_forward_hook(lambda *args:None)]
    before=(dict(model._forward_pre_hooks),dict(model._forward_hooks))
    monkeypatch.setattr(model,'register_forward_hook',fail)
    with pytest.raises(RuntimeError,match='injected registration failure'):
        AppendLease(model,Policy())
    assert (dict(model._forward_pre_hooks),dict(model._forward_hooks))==before
    for handle in existing:handle.remove()

@pytest.mark.parametrize('kind',[PromptQueryCapture,QuerySelector])
def test_layer_hook_close_is_idempotent(kind):
    attention=torch.nn.Linear(2,2)
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    lease=kind(model,Policy())
    assert len(attention._forward_pre_hooks)==1
    lease.close();lease.close()
    assert not attention._forward_pre_hooks and not lease.handles

@pytest.mark.parametrize('failed_layer',[0,1,2])
def test_snap_constructor_restores_groups_and_foreign_hooks(failed_layer,monkeypatch):
    attention=[torch.nn.Linear(2,2) for _ in range(3)]
    for i,a in enumerate(attention):a.num_key_value_groups=i+2
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=a) for a in attention]))
    existing=attention[0].register_forward_pre_hook(lambda *args:None)
    before=[dict(a._forward_pre_hooks) for a in attention]
    monkeypatch.setattr(attention[failed_layer],'register_forward_pre_hook',fail)
    with pytest.raises(RuntimeError,match='injected registration failure'):
        SnapKVAdapter(model)
    assert [dict(a._forward_pre_hooks) for a in attention]==before
    assert [a.num_key_value_groups for a in attention]==[2,3,4]
    existing.remove()

def test_snap_double_close_preserves_subsequent_group_change():
    attention=torch.nn.Linear(2,2);attention.num_key_value_groups=4
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    cache=SnapKVAdapter(model)
    assert attention.num_key_value_groups==1
    cache.close()
    assert attention.num_key_value_groups==4 and not attention._forward_pre_hooks
    attention.num_key_value_groups=8
    cache.close()
    assert attention.num_key_value_groups==8
