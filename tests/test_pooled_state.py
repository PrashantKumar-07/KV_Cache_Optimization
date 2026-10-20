"""Validate owned views, descriptor bounds and fail-before-dispatch guards."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from pooled_state import StateArena


def state(arena,layer):
    k=torch.zeros(2,8,4,dtype=torch.bfloat16);v=k.clone()
    cum,age,stamp=arena.attach(layer,torch.arange(8).float(),torch.arange(8).float(),torch.zeros(5))
    return SimpleNamespace(sram_k=k,sram_v=v,sram_cum=cum,sram_age=age)


def test_attach_preserves_roots_values_and_declared_owner_bytes():
    a=StateArena(3,16,20,'cpu');s=state(a,1)
    assert s.sram_cum.untyped_storage()._cdata==a.hot.untyped_storage()._cdata
    assert torch.equal(s.sram_cum,torch.arange(8).float()) and a.owned_gpu_bytes==(2*3*16+3*20)*4

@pytest.mark.parametrize('counts',[[[8,9,0]]*3,[[-1,0,1]]*3,[[2**63-1,2**63-1,2]]*3,[[1.,1.,1.]]*3,[[1,0,0]]])
def test_prepare_invalid_descriptors_stop_before_gpu_dispatch(counts):
    a=StateArena(3,16,20,'cpu');before=a.hot.clone()
    with pytest.raises(ValueError):a.prepare(counts)
    assert torch.equal(a.hot.view(torch.uint8),before.view(torch.uint8)) and a.descriptor_h2d_bytes==0

@pytest.mark.parametrize('fault',['negative','out_of_bounds','more_rows','duplicate_root','metadata_alias','bad_stride','oversized_group'])
def test_invalid_compaction_layout_alias_bounds_stop_before_dispatch(fault):
    a=StateArena(10,16,20,'cpu');states=[state(a,0),state(a,1)];keeps=[[0,2,4],[0,3,5]]
    if fault=='negative':keeps[1][0]=-1
    elif fault=='out_of_bounds':keeps[1][0]=8
    elif fault=='more_rows':keeps=[[0]*9,[0]*9]
    elif fault=='duplicate_root':states[1].sram_v=states[0].sram_k
    elif fault=='metadata_alias':states[1].sram_cum=states[0].sram_cum
    elif fault=='bad_stride':states[1].sram_k=states[1].sram_k.transpose(0,1)
    elif fault=='oversized_group':states=[state(a,i) for i in range(9)];keeps=[[0,2,4]]*9
    with pytest.raises(ValueError):a.compact(states,keeps,0)
    assert a.descriptor_h2d_bytes==a.scratch_peak_bytes==0
