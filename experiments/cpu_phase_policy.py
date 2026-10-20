"""Research decode composition with canonical native CPU state and GPU commands."""
import ast
import inspect
import textwrap
from functools import partial
from types import SimpleNamespace
import numpy as np
import torch
from cpu_state import CpuStateArena,kernel as cpu_kernel
from pooled_state_policy import PooledStatePolicy,PooledCache
from native_controller_policy import NativeControllerPolicy
from tiered_kv_cache import TieredKVCache
from shared_decode_transition import shared_contract
from probe_batched_eviction_selection import batched_victims

CPU_FIELDS={'sram_pos':('hot',False),'stt_pos':('cold',False),'_logical':('slots',False),'_free':('free',False),
            'sram_grace':('grace',True),'stt_shadow':('shadow',True),'stt_was_vram_resident':('resident',True),'stt_peek_streak':('streak',True)}


def cpu_property(name):
    plane,tensor=CPU_FIELDS[name]
    def get(self):
        arena=getattr(self,'_cpu_arena',None)
        if arena is None:return self.__dict__['_normal_'+name]
        version=int(arena.owners['version'][self._cpu_layer])
        if version!=self._cpu_cached_version:
            self._cpu_cached={};self._cpu_cached_version=version
        if name not in self._cpu_cached:
            view=arena.view(plane,self._cpu_layer)
            self._cpu_cached[name]=torch.from_numpy(view) if tensor else tuple(view.tolist())
        return self._cpu_cached[name]
    def set(self,value):
        if getattr(self,'_cpu_arena',None) is None:
            self.__dict__['_normal_'+name]=value
            return
        if name!='stt_peek_streak':raise RuntimeError('CPU state mutation bypassed native phase: '+name)
        target=get(self)
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=target.dtype or value.shape!=target.shape or value.requires_grad:raise ValueError('Invalid canonical streak update')
        if value.data_ptr()!=target.data_ptr():target.copy_(value)
    return property(get,set)


def reclaim_method():
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVCache.evict_and_demote)))
    body=tree.body[0].body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='n_over')
    # First n_over here is the cold overflow; hot selection is a separate method.
    end=next(i for i,n in enumerate(body[start:],start) if isinstance(n,ast.Expr) and ast.unparse(n)=='self.sram_grace.fill_(False)')
    copied=body[start:end]
    class NativeCommit(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0]) in ['self.stt_pos','self.stt_shadow','self.stt_was_vram_resident','self.stt_peek_streak']:return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            if ast.unparse(node.func)=='self.compact_stt_pair':node.func.attr='commit_cpu_reclaim'
            return self.generic_visit(node)
    module=ast.Module(body=[ast.FunctionDef(name='reclaim',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='no_backup_victims')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=ast.parse('cfg=self.cfg\ndropped=0\nreclaimed=0').body+copied+ast.parse('return dropped,reclaimed').body,decorator_list=[])],type_ignores=[])
    module=NativeCommit().visit(module)
    env={'torch':torch};exec(compile(ast.fix_missing_locations(module),'<original-cold-selection-native-commit>','exec'),env)
    return env['reclaim']
RECLAIM=reclaim_method()


class CpuPhaseCache(PooledCache):
    _cpu_arena=None
    _cpu_commands=None
    _cpu_promotion=None
    cpu_reclaims=0
    cpu_evictions=0
    for _name in CPU_FIELDS:locals()[_name]=cpu_property(_name)
    del _name

    def attach_cpu(self,arena,layer,policy):
        arena.load(layer,self.sram_pos,self.stt_pos,self.sram_grace.numpy(),self.stt_shadow.numpy(),self.stt_was_vram_resident.numpy(),self.stt_peek_streak.numpy(),self._logical,self._free)
        self._cpu_arena,self._cpu_layer,self._cpu_policy=arena,layer,policy
        self._cpu_cached_version=-1;self._cpu_cached={}

    def detach_cpu(self):
        if self._cpu_arena is None:return
        values={name:(getattr(self,name).clone() if tensor else list(getattr(self,name))) for name,(_,tensor) in CPU_FIELDS.items()}
        self._cpu_arena=None;self._cpu_cached={};self._cpu_commands=self._cpu_promotion=None
        for name,value in values.items():setattr(self,name,value)

    def promote(self,hits):
        if self._cpu_arena is None:return super().promote(hits)
        if self._cpu_promotion is None:raise RuntimeError('Unprepared CPU promotion')
        expected,count=self._cpu_promotion
        if hits!=expected:raise RuntimeError('CPU promotion identities changed')
        self._cpu_promotion=None
        return count

    def commit_cpu_reclaim(self,keep):
        arena=self._cpu_arena;layer=self._cpu_layer
        owners={name:a[layer:layer+1] for name,a in arena.owners.items()}
        cpu_kernel().reclaim(owners,np.asarray(keep,dtype=np.int32).reshape(1,-1),np.asarray([len(keep)],dtype=np.int32))
        arena.epoch+=1;self.cpu_reclaims+=1

    def evict_and_demote(self,new_k=None,new_v=None,new_pos=None):
        if self._cpu_arena is None:return super().evict_and_demote(new_k,new_v,new_pos)
        commands=self._cpu_commands;arena=self._cpu_arena;layer=self._cpu_layer
        if commands is None or commands.get('_owner') is not arena._command_owner or commands['versions'][layer]!=arena.owners['version'][layer]:raise RuntimeError('Stale native CPU layer commands')
        if new_k is None or new_v is None or new_k.shape[1]!=1 or new_pos!=self._cpu_new_pos:raise RuntimeError('CPU append differs from prepared phase')
        self.last_floor_protected=0
        # CPU positions/grace are already committed; append only GPU payload/metadata.
        self._extend_hot_pair(new_k.to(self.cfg.resolved_kv_dtype),new_v.to(self.cfg.resolved_kv_dtype))
        self.sram_cum=self.extend_metadata('sram_cum',1);self.sram_age=self.extend_metadata('sram_age',1)
        paid,restored,kept=map(int,commands['sizes'][layer])
        victim_rows=commands['paid'][layer,:paid].tolist();restored_rows=commands['restored'][layer,:restored].tolist()
        if paid:self._write(self.sram_k[:,victim_rows,:],self.sram_v[:,victim_rows,:],commands['writes'][layer,:paid].tolist())
        if restored:self.stt_last_access[restored_rows]=self._t
        if paid:self.stt_last_access=self.append_timestamps(paid)
        self.pending_keep=commands['keep'][layer,:kept]
        self._cpu_commands=None
        dropped=reclaimed=0
        if len(self.stt_pos)>self.cfg.sttram_capacity:
            # Reclaim uses compacted hot scores. Keep its original selection before
            # consuming cold keeps into native owners; KV writes were queued first.
            self._cpu_policy.state_arena.compact([self],[self.pending_keep],self._state_layer)
            self._cpu_policy.pooled_compact_groups+=1
            self.pending_keep=None
            dropped,reclaimed=RECLAIM(self,victim_rows)
        self.cpu_evictions+=1
        return paid+restored,dropped,restored,reclaimed

    def close(self):
        self.detach_cpu()
        super().close()


def packed(rows,dtype=np.int32):
    values=np.full((len(rows),max(map(len,rows),default=0)),-1,dtype=dtype)
    for i,row in enumerate(rows):values[i,:len(row)]=row
    return values,np.asarray(list(map(len,rows)),np.int32)


def protected_view(positions,cfg):
    if not len(positions):return torch.zeros(0,dtype=torch.bool)
    with np.errstate(over='ignore'):cutoff=positions.max()-np.int64(cfg.window_size)
    return torch.from_numpy((positions<cfg.sink_size)|(positions>cutoff))


class CpuPhasePolicy(PooledStatePolicy):
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls=partial(CpuPhaseCache,extra_slots=self.reserved_promotion_budget+self.max_decode_append)
        self.cpu_arena=None
        self.cpu_promote_calls=self.cpu_append_calls=self.cpu_demote_calls=self.cpu_fallback_steps=0

    def detach_cpu(self):
        for c in self.layer_caches.values():c.detach_cpu()
        self.cpu_arena=None

    def _visible_tuple(self,past):
        result=super()._visible_tuple(past)
        if self.cpu_arena is None and self.state_arena is not None and self.layer_caches:
            layers=list(self.layer_caches);caches=list(self.layer_caches.values())
            supported=layers==list(range(len(past))) and len({c.capacity for c in caches})==1 and all(c.stt_shadow.device.type=='cpu' and c.stt_peek_streak.dtype==torch.float32 for c in caches)
            if supported:
                arena=CpuStateArena(len(layers),self.state_arena.hot.shape[2],caches[0].capacity)
                for i,c in self.layer_caches.items():c.attach_cpu(arena,i,self)
                self.cpu_arena=arena
        return result

    def _prepare_decode_reads(self):
        self._pool_prepared=False
        for c in self.layer_caches.values():c._pool_prepared=False
        past=self.incoming_past;layers=list(self.layer_caches);arena=self.cpu_arena
        supported=bool(layers) and arena is not None and layers==list(range(len(past)))
        if supported:
            n_new=max(1,past[0][0].shape[2]-self._vis_len[layers[0]])
            roots=[t.untyped_storage()._cdata for pair in past for t in pair]
            supported=(n_new==1 and len(roots)==len(set(roots)) and
                len({(c.cfg.inclusive,c.cfg.stt_live_floor) for c in self.layer_caches.values()})==1 and
                len({c.metrics.prompt_len for c in self.layer_caches.values()})==1 and
                all(c._cpu_arena is arena and c.cfg.write_aware_lambda==0 and c.cfg.batch_backup_updates and
                    type(c.cfg.inclusive) is bool and type(c.cfg.stt_live_floor) is int and 0<=c.cfg.stt_live_floor<=np.iinfo(np.int32).max and
                    0<=c.cfg.sink_size<=np.iinfo(np.int64).max and 0<=c.cfg.window_size<=np.iinfo(np.int64).max and
                    len(c.sram_pos)+self.promotion_budget+1<=arena.owners['hot'].shape[1] and
                    len(c.stt_pos)+max(0,len(c.sram_pos)+self.promotion_budget+1-c.cfg.sram_capacity)<=c.capacity and
                    shared_contract(c,past[i],1,self.promotion_budget) for i,c in self.layer_caches.items()))
        if not supported:
            self.detach_cpu();self.cpu_fallback_steps+=1
            return super()._prepare_decode_reads()
        # Native promotion planning reads canonical CPU flags before any slot change.
        if not NativeControllerPolicy.native_prepare(self):
            self.detach_cpu();self.cpu_fallback_steps+=1
            return super()._prepare_decode_reads()
        caches=list(self.layer_caches.values());promotion_rows=[];old_hot=[]
        for i,c in self.layer_caches.items():
            if c._cpu_commands is not None or c._cpu_promotion is not None or c.pending_keep is not None:raise RuntimeError('Unconsumed native CPU phase')
            old_hot.append(len(c.sram_pos))
            hits,_=self._planned_promotions[i]
            rows=c.controller_rows[1] if c.controller_rows is not None else []
            if hits is not None:c.last_floor_deferred=c.controller_rows[2] if c.controller_rows is not None else 0
            c._cpu_promotion=(list(hits),len(rows)) if hits is not None else None
            promotion_rows.append(rows);c.controller_rows=None
            c._hot_buffer=past[i];self.shared_transition_bindings+=1;self.raw_bound_layers+=1
        gpu_counts=np.asarray([[n,len(rows),1] for n,rows in zip(old_hot,promotion_rows)],np.int32)
        self.state_arena.prepare(gpu_counts);self.pooled_prepare_calls+=1
        commands=arena.promote(*packed(promotion_rows),inclusive=caches[0].cfg.inclusive,floor=caches[0].cfg.stt_live_floor)
        self.cpu_promote_calls+=1;arena.validate_commands(commands)
        for i,c in self.layer_caches.items():
            rows=promotion_rows[i]
            if rows:
                keys,values=c.select_stt_pair(rows)
                c._extend_hot_pair(keys,values)
                c._pool_prepared=True
                c.sram_cum=c.extend_metadata('sram_cum',len(rows));c.sram_age=c.extend_metadata('sram_age',len(rows))
                if c.cfg.inclusive:c.stt_last_access[rows]=c._t+1
                else:
                    old_cold=len(c.stt_last_access);keep=[j for j in range(old_cold) if j not in set(rows)]
                    c.stt_last_access=c.stt_last_access[keep]
            c._pool_prepared=True
        base=caches[0].metrics.prompt_len+self.decode_step_count-1
        arena.append_hot(*packed([[base] for _ in caches],dtype=np.int64));self.cpu_append_calls+=1
        projections=[]
        for i,c in self.layer_caches.items():
            positions=arena.view('hot',i);n=len(positions)
            projections.append(SimpleNamespace(cfg=c.cfg,sram_pos=positions,sram_cum=self.state_arena.hot[0,i,:n],sram_age=self.state_arena.hot[1,i,:n],sram_grace=c.sram_grace,
                _hard_protected_mask=lambda pos=positions,cfg=c.cfg:protected_view(pos,cfg)))
        selected=batched_victims(projections,numpy_masks=False)
        commands=arena.demote(*packed(selected),inclusive=caches[0].cfg.inclusive);self.cpu_demote_calls+=1
        for c in caches:c._cpu_commands=commands;c._cpu_new_pos=base
        self._pool_prepared=True

    def reset(self):
        super().reset()
        self.cpu_arena=None
        self.cpu_promote_calls=self.cpu_append_calls=self.cpu_demote_calls=self.cpu_fallback_steps=0
