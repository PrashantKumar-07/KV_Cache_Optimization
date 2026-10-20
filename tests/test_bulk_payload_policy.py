import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
from bulk_payload_policy import BulkPayloadPool,BulkPayloadCache

def pool():
    p=BulkPayloadPool.__new__(BulkPayloadPool);p.pending=[];p.bulk_pending=None;p.dim=2;p.device=torch.device('cpu');p.pool=torch.zeros(2,8,1,2,dtype=torch.bfloat16)
    c=SimpleNamespace(manager=p,base=0,capacity=4,cfg=SimpleNamespace(num_heads=2),host_write_bytes=0,d2h_calls=0);p.caches={0:c};return p,c

def test_bulk_queue_retains_tensor_and_bills_exact_submitted_bytes():
    p,c=pool();data=torch.zeros(2,4,1,2,dtype=torch.bfloat16);p.queue_bulk(data,[(c,[1,3])]);assert p.bulk_pending[0] is data;assert p.bulk_pending[2]==[1,3,5,7];assert c.host_write_bytes==32

@pytest.mark.parametrize('case',['duplicates','noninteger','foreign'])
def test_invalid_physical_queue_rejects_without_counter_or_queue_mutation(case):
    p,c=pool();slots=[1,1] if case=='duplicates' else [1,3]
    if case=='noninteger':slots=[1.5,3]
    if case=='foreign':c.manager=object()
    with pytest.raises(ValueError):p.queue_bulk(torch.zeros(2,4,1,2,dtype=torch.bfloat16),[(c,slots)])
    assert p.bulk_pending is None and c.host_write_bytes==0

def test_zero_paid_does_not_queue_or_bill_dispatch():
    p,c=pool();p.queue_bulk(torch.empty(2,0,1,2,dtype=torch.bfloat16),[(c,[])]);assert p.bulk_pending is None and c.host_write_bytes==0

def test_pending_writes_cannot_be_overwritten():
    p,c=pool();p.pending=[object()]
    with pytest.raises(RuntimeError):p.queue_bulk(torch.zeros(2,4,1,2,dtype=torch.bfloat16),[(c,[1,3])])
    assert p.bulk_pending is None and c.host_write_bytes==0

class Cache(BulkPayloadCache):
    def __del__(self):pass

def cache():
    c=Cache.__new__(Cache);owner=object();c._cpu_arena=SimpleNamespace(_command_owner=owner,owners={'version':np.array([7],dtype=np.int64)});c._cpu_layer=0;c._cpu_new_pos=23;c._bulk_payload_ready=True;c.cpu_evictions=0
    c._cpu_commands={'_owner':owner,'versions':np.array([7],dtype=np.int64),'sizes':np.array([[2,3,4]],dtype=np.int32)};return c

def test_native_cpu_commands_consumed_once_with_original_counts():
    c=cache();kv=torch.zeros(2,1,4);assert c.evict_and_demote(kv,kv,new_pos=23)==(5,0,3,0);assert c._cpu_commands is None and not c._bulk_payload_ready and c.cpu_evictions==1

@pytest.mark.parametrize('case',['stale','wrong_append'])
def test_bulk_command_errors_preserve_unconsumed_phase(case):
    c=cache();command=c._cpu_commands
    if case=='stale':command['versions'][0]=6
    with pytest.raises(RuntimeError):c.evict_and_demote(torch.zeros(2,1,4),torch.zeros(2,1,4),new_pos=24 if case=='wrong_append' else 23)
    assert c._cpu_commands is command and c._bulk_payload_ready and c.cpu_evictions==0
