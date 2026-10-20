"""Exact ragged projected scores and original stable victim/fallback selection."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from batched_eviction_projection import projected_scores,PROJECTED_VICTIMS,ProjectedEvictionPolicy,COORDINATOR_AST
from probe_batched_eviction_selection import batched_victims,cache
from bulk_promotion_policy import BulkCache


def bits(t):return t.contiguous().view(torch.int32)


@pytest.mark.parametrize('n_new',[1,4,32])
@pytest.mark.parametrize('case',['ties','grace','nonfinite','subnormal','protected','nooverflow'])
def test_ragged_projection_preserves_exact_scores_and_victims(case,n_new):
    torch.manual_seed(1);original=[];projections=[]
    for length in [0,1,9,33,64]:
        cum=torch.randint(0,8,(length,)).float();age=torch.randint(0,8,(length,)).float()
        if case=='nonfinite' and length>=9: cum[4:9]=torch.tensor([float('nan'),float('inf'),-float('inf'),-0.,0.])
        if case=='subnormal':cum.fill_(torch.nextafter(torch.tensor(0.),torch.tensor(1.)))
        positions=list(range(length))+list(range(1000,1000+n_new))
        grace=torch.zeros(length+n_new,dtype=torch.bool)
        if case=='grace':grace[:length]=True
        capacity=len(positions) if case=='nooverflow' else max(0,length-3)
        common=dict(capacity=capacity,grace=grace,window=2000 if case=='protected' else 2,sink=0)
        original.append(cache(positions,cum,age,**common))
        projections.append(cache(positions,torch.cat((cum,torch.zeros(n_new))),torch.cat((age+1,torch.zeros(n_new))),**common))
    cumulative,ages=projected_scores(original,n_new)
    for i,c in enumerate(projections):
        assert torch.equal(bits(cumulative[i,:len(c.sram_pos)]),bits(c.sram_cum))
        assert torch.equal(bits(ages[i,:len(c.sram_pos)]),bits(c.sram_age))
    assert PROJECTED_VICTIMS(original,numpy_masks=True,n_new=n_new)==batched_victims(projections,numpy_masks=True)


def test_policy_keeps_same_bulk_cache_and_only_projection_coordinator_changes():
    policy=ProjectedEvictionPolicy(sram_budget=991,stt_budget=1090,start_size=4,recent_size=128)
    try:
        assert policy.direct_cache_cls.func is BulkCache
        assert sorted(COORDINATOR_AST['changes'])==['selection','sram_age','sram_cum','unusedzeros']
    finally:policy.reset()
