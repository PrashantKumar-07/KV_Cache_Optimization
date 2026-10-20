"""Validate every shared hotprefix before alias writes; preserve fallback."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from shared_decode_transition import shared_contract,SharedTransitionPolicy,SharedAppendLease
from bulk_promotion_policy import BulkPromotionPolicy
from raw_input_reuse import RawInputPolicy


def fixture():
    key=torch.zeros(1,2,16,4,dtype=torch.bfloat16);value=key.clone()
    cache=SimpleNamespace(sram_pos=list(range(10)),sram_k=key[0,:,:10],sram_v=value[0,:,:10],_hot_buffer=None)
    return cache,(key,value)


@pytest.mark.parametrize('fault',['none','multi','short','copied','alias','dtype','stride'])
def test_sharedprefix_contract_rejects_unsupported_geometry(fault):
    cache,pair=fixture();n_new=1
    if fault=='multi':n_new=2
    elif fault=='short':pair=(pair[0][:,:,:12],pair[1][:,:,:12])
    elif fault=='copied':cache.sram_k=cache.sram_k.clone()
    elif fault=='alias':pair=(pair[0],pair[0]);cache.sram_v=cache.sram_k
    elif fault=='dtype':pair=(pair[0].float(),pair[1].float())
    elif fault=='stride':cache.sram_k=cache.sram_k.transpose(0,1)
    assert shared_contract(cache,pair,n_new,4)==(fault=='none')


def test_crosslayer_preparation_binds_and_clears_on_error(monkeypatch):
    p=SharedTransitionPolicy(sram_budget=991,stt_budget=1090,promotion_budget=4,start_size=4,recent_size=128)
    entries=[fixture(),fixture()];p.layer_caches={i:e[0] for i,e in enumerate(entries)};p.incoming_past=tuple(e[1] for e in entries);p._vis_len={0:15,1:15}
    def fail(self):
        assert all(c._hot_buffer is self.incoming_past[i] for i,c in self.layer_caches.items())
        raise ValueError('injected prepare failure')
    monkeypatch.setattr(BulkPromotionPolicy,'_prepare_decode_reads',fail)
    with pytest.raises(ValueError,match='injected'):p._prepare_decode_reads()
    assert p.shared_transition_bindings==2 and all(c._hot_buffer is None for c in p.layer_caches.values())
    p.layer_caches={};p.manager=None;p.reset()


def test_invalid_layer_alias_delegates_before_shared_mutations(monkeypatch):
    p=SharedTransitionPolicy(sram_budget=991,stt_budget=1090,promotion_budget=4,start_size=4,recent_size=128)
    cache,pair=fixture();p.layer_caches={0:cache,1:cache};p.incoming_past=(pair,pair);p._vis_len={0:15,1:15};called=[]
    monkeypatch.setattr(RawInputPolicy,'_prepare_decode_reads',lambda self:called.append(True))
    p._prepare_decode_reads();assert called and p.shared_transition_bindings==0 and cache._hot_buffer is None
    p.layer_caches={};p.manager=None;p.reset()


def test_external_instance_update_is_preserved_without_lease():
    from transformers.cache_utils import DynamicLayer
    layer=DynamicLayer();custom=lambda *args:None;layer.update=custom
    policy=type('Policy',(),{'initialized':True})();model=torch.nn.Identity();model.config=SimpleNamespace(model_type='mistral')
    lease=SharedAppendLease(model,policy)
    lease.before(model,(),dict(past_key_values=SimpleNamespace(layers=[layer]),input_ids=torch.ones(1,1),use_cache=True))
    assert not lease.pending and layer.update is custom
    lease.close()


def test_immediate_compaction_preserves_exact_payload_metadata_and_root():
    from shared_decode_transition import SharedTransitionCache
    from raw_input_reuse import RawInputCache
    from types import MethodType
    key=torch.arange(2*16*4,dtype=torch.int16).reshape(1,2,16,4).view(torch.bfloat16).clone();value=key.clone()
    cache=SimpleNamespace(sram_k=key[0],sram_v=value[0],sram_pos=list(range(16)),sram_grace=torch.tensor([i%2==0 for i in range(16)]),sram_cum=torch.arange(16).float(),sram_age=torch.arange(16).float()/2,pending_keep=None,_hot_buffer=(key,value),shared_immediate_compactions=0,reused_compactions=0)
    cache._supported_hot_layout=RawInputCache._supported_hot_layout
    victims=[2,6,12];keep=[i for i in range(16) if i not in victims]
    expected=[t.index_select(1,torch.tensor(keep)).clone() for t in (cache.sram_k,cache.sram_v)]
    cum=cache.sram_cum[keep].clone();age=cache.sram_age[keep].clone();grace=cache.sram_grace[keep].clone()
    roots=[t.untyped_storage()._cdata for t in (key,value)]
    SharedTransitionCache._compact_or_queue(cache,victims)
    assert cache.pending_keep is None and cache.shared_immediate_compactions==1
    assert cache.sram_pos==keep and torch.equal(cache.sram_cum,cum) and torch.equal(cache.sram_age,age) and torch.equal(cache.sram_grace,grace)
    for output,reference,root in zip((cache.sram_k,cache.sram_v),expected,roots):
        assert torch.equal(output.contiguous().view(torch.int16),reference.contiguous().view(torch.int16))
        assert output.untyped_storage()._cdata==root
