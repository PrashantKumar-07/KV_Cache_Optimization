"""Payload/metadata preservation and safe overlapping gathers on CPU fixtures."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import raw_input_reuse as reuse


def bits(t): return t.contiguous().view(torch.uint8)


def fixture():
    raw = tuple(torch.arange(2*16*8,dtype=torch.int16).reshape(1,2,16,8).view(torch.bfloat16).clone()
                for _ in range(2))
    cache = SimpleNamespace(sram_k=raw[0][0,:,:6].clone(),sram_v=raw[1][0,:,:6].clone(),
        sram_pos=list(range(6)),sram_cum=torch.arange(6,dtype=torch.float32),
        sram_age=torch.arange(6,dtype=torch.float32),sram_grace=torch.zeros(6,dtype=torch.bool),
        stt_pos=list(range(6,14)),stt_shadow=torch.zeros(8,dtype=torch.bool),
        stt_was_vram_resident=torch.zeros(8,dtype=torch.bool),stt_last_access=torch.zeros(8),
        cfg=SimpleNamespace(device='cpu',dtype=torch.float32,resolved_kv_dtype=torch.bfloat16,inclusive=True),
        _t=3,_hot_buffer=None)
    return cache,raw


def test_original_metadata_and_payload_survive_promote_append_hooks():
    control,raw = fixture(); candidate=copy.deepcopy(control)
    keys=torch.randn(2,2,8).bfloat16();values=keys.flip(2).clone()
    for cache in (control,candidate):
        cache.promotion_rows=lambda hits:([1,3],0)
        cache.select_stt_pair=lambda idx:(keys,values)
    candidate._hot_buffer=raw
    candidate.sram_k,candidate.sram_v=(t[0,:,:6] for t in raw)
    candidate._extend_hot_pair=lambda k,v:reuse.RawInputCache._extend_hot_pair(candidate,k,v)
    assert reuse.TieredKVCache.promote(control,[(1,2)])==reuse.PROMOTE(candidate,[(1,2)])==2
    new_k=torch.randn(2,1,8);new_v=new_k.flip(2)
    reuse.TieredKVCache.append_tokens(control,new_k,new_v,14)
    reuse.APPEND(candidate,new_k,new_v,14)
    for name in ('sram_k','sram_v','sram_cum','sram_age','sram_grace','stt_shadow','stt_was_vram_resident','stt_last_access'):
        assert torch.equal(bits(getattr(control,name)),bits(getattr(candidate,name))),name
    assert control.sram_pos==candidate.sram_pos and candidate.sram_k.data_ptr()==raw[0].data_ptr()


@pytest.mark.parametrize('keep',([0,1,2,3],[5,0,3,3],[],[5]))
def test_full_scratch_gather_handles_overlapping_rows_duplicates_and_empty(keep):
    cache,raw=fixture();cache._hot_buffer=raw
    cache.sram_k,cache.sram_v=(t[0,:,:6] for t in raw)
    cache.pending_keep=keep
    index=torch.tensor(keep,dtype=torch.long)
    expected={n:getattr(cache,n).index_select(1 if n in ('sram_k','sram_v') else 0,index)
              for n in ('sram_k','sram_v','sram_cum','sram_age')}
    tails=[t[:,:,len(keep):].clone() for t in raw]
    reuse.gather_in_place(cache)
    for name,want in expected.items():assert torch.equal(bits(getattr(cache,name)),bits(want))
    assert all(torch.equal(bits(t[:,:,len(keep):]),bits(old)) for t,old in zip(raw,tails))


@pytest.mark.parametrize('fault',('prefix','dtype','short','alias','multiappend'))
def test_invalid_binding_preserves_buffers(fault):
    cache,raw=fixture();n_new=1
    assert reuse.binding_contract(cache,raw,1,2)
    if fault=='prefix':raw[0][0,0,0,0]=42
    elif fault=='dtype':raw=tuple(t.float() for t in raw)
    elif fault=='short':raw=tuple(t[:,:,:8].contiguous() for t in raw)
    elif fault=='alias':raw=(raw[0],raw[0])
    else:n_new=2
    before=[t.clone() for t in raw]
    assert not reuse.binding_contract(cache,raw,n_new,2)
    assert all(torch.equal(bits(t),bits(old)) for t,old in zip(raw,before))


def test_aliasing_extension_falls_back_before_overwriting_source():
    cache,raw=fixture();cache._hot_buffer=raw
    cache.sram_k,cache.sram_v=(t[0,:,:6] for t in raw)
    keys,values=(t[0,:,2:4] for t in raw)
    expected=[torch.cat((getattr(cache,n),v),dim=1) for n,v in (('sram_k',keys),('sram_v',values))]
    before=[t.clone() for t in raw]
    reuse.RawInputCache._extend_hot_pair(cache,keys,values)
    assert cache._hot_buffer is None
    assert all(torch.equal(bits(getattr(cache,n)),bits(v)) for n,v in zip(('sram_k','sram_v'),expected))
    assert all(torch.equal(bits(t),bits(old)) for t,old in zip(raw,before))
