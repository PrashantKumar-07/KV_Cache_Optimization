"""Research CPU phase composition with a checked bounded GPU payload executor."""
import ast,inspect,textwrap
from functools import partial
from types import SimpleNamespace
import torch
from bulk_payload import execute
from cpu_phase_policy import CpuPhaseCache,CpuPhasePolicy
from native_controller_policy import ControllerPool
from pooled_state_policy import PooledDecode,decode_transform
from longbench_eval import TieredKVPolicy

class BulkPayloadPool(ControllerPool):
    def __init__(self,*args):
        super().__init__(*args);self.bulk_pending=None;self.bulk_dispatches=0
    def queue_bulk(self,packed,rows):
        if self.pending or self.bulk_pending is not None:raise RuntimeError('Unconsumed physical writes before bulk queue')
        expected=sum(c.cfg.num_heads*len(slots) for c,slots in rows)
        if packed.shape!=(2,expected,1,self.dim) or packed.dtype!=self.pool.dtype or packed.device!=self.device:raise ValueError('Invalid bulk write payload')
        if any(c.manager is not self or id(c) not in {id(v) for v in self.caches.values()} for c,_ in rows):raise ValueError('Foreign bulk cache')
        if any(type(slot) is not int for _,slots in rows for slot in slots):raise ValueError('Invalid bulk physical slot type')
        addresses=[c.base+h*c.capacity+int(slot) for c,slots in rows for h in range(c.cfg.num_heads) for slot in slots]
        if any(not 0<=int(slot)<c.capacity for c,slots in rows for slot in slots) or len(set(addresses))!=len(addresses) or any(not 0<=a<self.pool.shape[1] for a in addresses):raise ValueError('Invalid bulk physical slots')
        if not expected:return
        self.bulk_pending=(packed,rows,addresses)
        for c,slots in rows:c.host_write_bytes+=len(slots)*c.cfg.num_heads*2*self.dim*self.pool.element_size()
    def enqueue(self,*args):
        if self.bulk_pending is not None:self.flush()
        return super().enqueue(*args)
    def flush(self):
        super().flush()
        if self.bulk_pending is None:return
        packed,rows,addresses=self.bulk_pending
        with torch.cuda.device(self.device):
            self.wait();destination=torch.tensor(addresses,dtype=torch.int32,device=self.device)
            source=torch.arange(packed.shape[1],dtype=torch.int32,device=self.device)
            packed.record_stream(torch.cuda.current_stream(self.device))
            self.d2h.kv_cache_d2h_transfer(packed,self.pool,source,destination)
            self.record();self.d2h_calls+=1;self.bulk_dispatches+=1
            for c,slots in rows:
                if slots:c.d2h_calls+=1
            self.bulk_pending=None

class BulkPayloadCache(CpuPhaseCache):
    _bulk_payload_ready=False
    def evict_and_demote(self,new_k=None,new_v=None,new_pos=None):
        if not self._bulk_payload_ready:return super().evict_and_demote(new_k,new_v,new_pos)
        command=self._cpu_commands;arena=self._cpu_arena;layer=self._cpu_layer
        if command is None or command.get('_owner') is not arena._command_owner or command['versions'][layer]!=arena.owners['version'][layer]:raise RuntimeError('Stale bulk CPU commands')
        if new_k is None or new_v is None or new_k.shape[1]!=1 or new_pos!=self._cpu_new_pos:raise RuntimeError('Bulk append differs from prepared phase')
        paid,restored,_=map(int,command['sizes'][layer]);self.last_floor_protected=0
        self._cpu_commands=None;self._bulk_payload_ready=False;self.cpu_evictions+=1
        return paid+restored,0,restored,0

def decode_method():
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVPolicy._decode_step)));decode_transform(tree)
    count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['new_k','new_v']:
            assert isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='to'
            node.value=ast.IfExp(ast.parse('self.bulk_payload_applied',mode='eval').body,node.value.func.value,node.value);count+=1
    assert count==2
    env=TieredKVPolicy._decode_step.__globals__.copy();exec(compile(ast.fix_missing_locations(tree),'<bulk-payload-original-decode>','exec'),env)
    return env['_decode_step']

class BulkDecode(PooledDecode):
    _decode_step=decode_method()

class BulkPayloadPolicy(CpuPhasePolicy,BulkDecode):
    direct_pool_cls=BulkPayloadPool
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls=partial(BulkPayloadCache,extra_slots=self.reserved_promotion_budget+self.max_decode_append)
        self.bulk_payload_applied=False;self.bulk_payload_steps=self.bulk_payload_fallback_steps=0
        self.bulk_descriptor_h2d_bytes=self.bulk_scratch_peak_bytes=self.bulk_paid_peak_bytes=0
    def _prepare_decode_reads(self):
        self.bulk_payload_applied=False
        for c in self.layer_caches.values():c._bulk_payload_ready=False
        super()._prepare_decode_reads()
        caches=list(self.layer_caches.values());past=self.incoming_past;arena=self.state_arena
        if (not caches or self.cpu_arena is None or not self._pool_prepared or self.manager.pending or self.manager.bulk_pending is not None or
            any(c._cpu_commands is None or not c.cfg.inclusive or len(c.stt_pos)>c.cfg.sttram_capacity for c in caches)):
            self.bulk_payload_fallback_steps+=1;return
        states=[];keeps=[];paid=[];restored=[];writes=[]
        for i,c in enumerate(caches):
            command=c._cpu_commands;self.cpu_arena.validate_commands(command)
            p,r,k=map(int,command['sizes'][i]);old=c.sram_k.shape[1];cold=len(c.stt_last_access)
            states.append(SimpleNamespace(old_k=c.sram_k,old_v=c.sram_v,prom_k=c.sram_k[:,:0],prom_v=c.sram_v[:,:0],new_k=past[i][0][0,:,-1:,:],new_v=past[i][1][0,:,-1:,:],cumulative=arena.hot[0,i,:old+1],age=arena.hot[1,i,:old+1],timestamps=arena.cold[i,:cold+p],old_cold=cold,time=c._t+1))
            keeps.append(command['keep'][i,:k]);paid.append(command['paid'][i,:p]);restored.append(command['restored'][i,:r]);writes.append((c,command['writes'][i,:p].tolist()))
        try:result=execute(states,keeps,paid,restored)
        except ValueError:
            self.bulk_payload_fallback_steps+=1;return
        self.manager.queue_bulk(result['paid'],writes)
        for c,state,views in zip(caches,states,result['views']):
            c.sram_k,c.sram_v,c.sram_cum,c.sram_age=views;c.stt_last_access=state.timestamps
            c.pending_keep=None;c._bulk_payload_ready=True;c.batched_deferred_compactions+=1;c.reused_compactions=getattr(c,'reused_compactions',0)+1
        self.pooled_compact_groups+=(len(caches)+7)//8
        self.bulk_payload_applied=True;self.bulk_payload_steps+=1
        self.bulk_descriptor_h2d_bytes+=result['descriptor_h2d_bytes'];self.bulk_scratch_peak_bytes=max(self.bulk_scratch_peak_bytes,result['scratch_peak_bytes']);self.bulk_paid_peak_bytes=max(self.bulk_paid_peak_bytes,result['paid_buffer_bytes'])
    def reset(self):
        super().reset();self.bulk_payload_applied=False
        self.bulk_payload_steps=self.bulk_payload_fallback_steps=0
        self.bulk_descriptor_h2d_bytes=self.bulk_scratch_peak_bytes=self.bulk_paid_peak_bytes=0
