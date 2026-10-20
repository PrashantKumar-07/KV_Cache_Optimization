"""Regression proof for consumed-cache output, aliases and extracted visibility."""
import ast
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import textwrap
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import reuse_visible_policy as reuse


def fixture():
    bits=torch.arange(8*16*128,dtype=torch.int32).to(torch.int16)
    state=SimpleNamespace(sram_k=bits.view(torch.bfloat16).reshape(8,16,128),
        sram_v=bits.flip(0).view(torch.bfloat16).reshape(8,16,128),
        sram_cum=torch.arange(16,dtype=torch.float32),sram_age=torch.arange(16,dtype=torch.float32),
        pending_keep=np.asarray([9,0,3,3],dtype=np.int32))
    raw=tuple(torch.full((1,8,16,128),-777,dtype=torch.bfloat16) for _ in range(2))
    return state,raw


def bytes(t):return t.contiguous().view(torch.uint8)


def test_only_two_concat_assignments_change_in_production_visibility():
    tree=ast.parse(textwrap.dedent(inspect.getsource(reuse.TieredKVPolicy._visible_tuple)))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('vis_k','vis_v'):
            c=node.targets[0].id[-1]
            node.value=ast.parse(f'self._compose_visible(fast_{c}, cold_{c}, {c})',mode='eval').body
    assert ast.dump(ast.fix_missing_locations(tree))==reuse.REUSE_AST


def test_exact_hot_gather_cold_copy_and_untouched_tail():
    state,raw=fixture();indices=torch.tensor(state.pending_keep,dtype=torch.long)
    expected={name:getattr(state,name).index_select(1 if name in ('sram_k','sram_v') else 0,indices)
        for name in ('sram_k','sram_v','sram_cum','sram_age')}
    original=[t.clone() for t in raw]
    assert reuse.can_reuse([state],[raw],4)
    reuse.gather_into([state],[raw])
    for name,want in expected.items():assert torch.equal(bytes(getattr(state,name)),bytes(want))
    for channel,name in enumerate(('sram_k','sram_v')):
        fast=getattr(state,name);cold=torch.full((8,4,128),channel+12,dtype=torch.bfloat16)
        result=reuse.ReuseVisiblePolicy._compose_visible(fast,cold,raw[channel])
        assert result.untyped_storage()._cdata==raw[channel].untyped_storage()._cdata
        assert torch.equal(bytes(result),bytes(torch.cat((expected[name],cold),dim=1)))
        assert torch.equal(bytes(raw[channel][:,:,8:]),bytes(original[channel][:,:,8:]))


@pytest.mark.parametrize('fault',('source_alias','cross_source_alias','metadata_alias','outputs_alias',
    'shape','dtype','bounds','negative','ragged','noncontiguous'))
def test_rejects_unsafe_destination_geometry_before_any_write(fault):
    a,raw_a=fixture();b,raw_b=fixture();states=[a,b];raw=[raw_a,raw_b]
    if fault=='source_alias':a.sram_k=raw_a[0][0]
    elif fault=='cross_source_alias':b.sram_k=raw_a[0][0]
    elif fault=='metadata_alias':a.sram_cum=raw_a[0].view(torch.float32).view(-1)[:16]
    elif fault=='outputs_alias':raw[1]=raw_a
    elif fault=='shape':a.sram_v=a.sram_v[:,:15]
    elif fault=='dtype':a.sram_age=a.sram_age.to(torch.float64)
    elif fault=='bounds':a.pending_keep[0]=16
    elif fault=='negative':a.pending_keep[0]=-1
    elif fault=='ragged':b.pending_keep=b.pending_keep[:-1]
    else:raw[0]=tuple(t.transpose(1,2) for t in raw_a)
    before=[t.clone() for pair in raw for t in pair]
    assert not reuse.can_reuse(states,raw,4)
    assert all(torch.equal(bytes(t),bytes(old)) for t,old in zip([t for pair in raw for t in pair],before))


def test_same_storage_wrong_head_stride_falls_back_to_concat():
    _,raw=fixture();fast=raw[0].view(-1)[:8*4*128].view(8,4,128)
    cold=torch.ones(8,4,128,dtype=torch.bfloat16)
    result=reuse.ReuseVisiblePolicy._compose_visible(fast,cold,raw[0])
    assert result.untyped_storage()._cdata!=raw[0].untyped_storage()._cdata
    assert torch.equal(bytes(result),bytes(torch.cat((fast,cold),dim=1)))


def test_raw_storage_offset_preserved_by_reused_views():
    state,_=fixture()
    parents=tuple(torch.full((2,8,16,128),-999,dtype=torch.bfloat16) for _ in range(2))
    raw=tuple(t[1:2] for t in parents)
    assert reuse.can_reuse([state],[raw],4)
    expected=[t.index_select(1,torch.tensor(state.pending_keep,dtype=torch.long))
        for t in (state.sram_k,state.sram_v)]
    reuse.gather_into([state],[raw])
    for fast,want,original,parent in zip((state.sram_k,state.sram_v),expected,raw,parents):
        assert fast.storage_offset()==original.storage_offset()
        assert torch.equal(bytes(fast),bytes(want))
        assert torch.all(parent[0]==-999)
