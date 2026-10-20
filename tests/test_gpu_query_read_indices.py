"""PureCPU address-map proof before unchecked native kernel dispatch."""
from types import SimpleNamespace
import sys
from pathlib import Path
import numpy as np
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from gpu_query_read_indices import source_maps,version


def caches():
    return {i:SimpleNamespace(cfg=SimpleNamespace(num_heads=2),capacity=8,base=i*16,
        _logical=[5,1,7,3],stt_pos=[11,12,13,14],stt_shadow=torch.tensor([False,True,False,False])) for i in range(3)}


def test_batched_source_map_gaps_shadow_and_layer_offsets():
    data=caches();blocks,live,lengths,versions=source_maps(data)
    assert lengths==[3,3,3] and versions==[version(c) for c in data.values()]
    for layer in range(3):
        assert live[layer,:3].tolist()==[0,2,3]
        for head in range(2):assert blocks[layer,head,:3].tolist()==[layer*16+head*8+i for i in [5,7,3]]
    assert blocks.dtype==np.int32 and live.dtype==np.int64


@pytest.mark.parametrize('bad',['duplicate','negative','overflow','position_length','shadow_length','base','heads'])
def test_invalid_sources_rejected_before_native_dispatch(bad):
    data=caches();c=data[1]
    if bad=='duplicate':c._logical[0]=c._logical[1]
    elif bad=='negative':c._logical[0]=-1
    elif bad=='overflow':c._logical[0]=8
    elif bad=='position_length':c.stt_pos.pop()
    elif bad=='shadow_length':c.stt_shadow=c.stt_shadow[:2]
    elif bad=='base':c.base=1
    else:c.cfg.num_heads=3
    with pytest.raises(ValueError,match='Invalid native'):source_maps(data)


def test_full_shadow_and_reordered_slots_refresh():
    data=caches();data[0].stt_shadow[:]=True;before=version(data[1]);data[1]._logical.reverse()
    blocks,live,lengths,_=source_maps(data);assert lengths[0]==0 and version(data[1])!=before
    assert blocks[1,0,:3].tolist()==[16+3,16+1,16+5]


def test_int32_source_address_overflow_rejected_before_allocation():
    data=caches()
    for cache in data.values():cache.capacity=2**30
    with pytest.raises(ValueError,match='int32 geometry'):source_maps(data)
