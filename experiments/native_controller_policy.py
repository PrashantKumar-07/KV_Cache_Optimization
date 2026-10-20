"""Research controller: native CPU plans/addresses, grouped GPU timestamp rank."""
import ast
from functools import partial
import inspect
import math
import textwrap
from types import SimpleNamespace
import numpy as np
import torch
from native_controller import kernel
from bounded_deferred_ownership import DeferredOwnershipPolicy,DeferredOwnershipCache
from bulk_promotion_policy import BulkPrepare
from direct_shared_pool import DirectSharedPool
from cpu_batched_cache import SharedPool
from fused_hot_compaction import gather
from reuse_visible_policy import ReuseVisiblePolicy
from longbench_eval import TieredKVPolicy


def address_tensor(manager,selections):
    frames=[dict(heads=c.cfg.num_heads,capacity=c.capacity,base=c.base,slots=[c._logical[i] for i in indices])
            for layer,indices in selections.items() for c in [manager.caches[layer]]]
    values=kernel().addresses(frames)
    manager.controller_address_calls+=1;manager.controller_address_array_peak=max(manager.controller_address_array_peak,values.nbytes)
    return torch.as_tensor(values,dtype=torch.int32,device=manager.device)


def read_method():
    tree=ast.parse(textwrap.dedent(inspect.getsource(SharedPool.read)))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='source')
    assert ast.unparse(node.value.func)=='torch.tensor'
    node.value=ast.parse('address_tensor(self,selections)',mode='eval').body
    env=SharedPool.read.__globals__.copy();env['address_tensor']=address_tensor
    exec(compile(ast.fix_missing_locations(tree),'<controller-read-addresses>','exec'),env)
    return env['read']


class ControllerPool(DirectSharedPool):
    read=read_method()
    def __init__(self,*args):
        super().__init__(*args);self.controller_address_calls=self.controller_address_array_peak=0


def eviction_method():
    from tiered_kv_cache import TieredKVCache
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVCache.evict_and_demote)))
    body=tree.body[0].body
    compact=next(n for n in body if isinstance(n,ast.If) and ast.unparse(n.test)=='victims')
    compact.body=ast.parse('self._compact_or_queue(victims)').body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='backup_by_pos')
    end=next(i for i,n in enumerate(body[start:],start) if isinstance(n,ast.If) and ast.unparse(n.test)=='restored_backups')
    body[start:end]=ast.parse('no_backup_victims,restored_backups=self.native_backups(victims)\ndemoted=len(victims)\nwrites_saved=len(restored_backups)').body
    env={'torch':torch}
    exec(compile(ast.fix_missing_locations(tree),'<controller-original-eviction>','exec'),env)
    return env['evict_and_demote']
NATIVE_EVICT=eviction_method()


class ControllerCache(DeferredOwnershipCache):
    controller_rows=None
    controller_backup_calls=0
    def native_backups(self,victims):
        result=kernel().backups([dict(hot=self.sram_pos,cold=self.stt_pos,shadow=self.stt_shadow.numpy(),victims=victims,inclusive=self.cfg.inclusive)])[0]
        self.controller_backup_calls+=1
        return result['paid'],result['restored']
    def evict_and_demote(self,*args,**options):
        if not self.cfg.batch_backup_updates or self.stt_shadow.device.type!='cpu' or self.stt_shadow.dtype!=torch.bool or not self.stt_shadow.is_contiguous():
            return super().evict_and_demote(*args,**options)
        return NATIVE_EVICT(self,*args,**options)
    def promotion_rows(self,hits):
        if self.controller_rows is None:return super().promotion_rows(hits)
        expected,rows,deferred=self.controller_rows
        if hits!=expected:raise RuntimeError('Controller promotion identities changed before consumption')
        self.controller_rows=None
        return rows,deferred


def visibility_method():
    tree=ast.parse(textwrap.dedent(inspect.getsource(TieredKVPolicy._visible_tuple)))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='live_by_layer')
    node.value=ast.parse('self.controller_live(past_kv)',mode='eval').body
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['vis_k','vis_v']:
            channel=node.targets[0].id[-1]
            assert ast.unparse(node.value)==f'torch.cat([fast_{channel}, cold_{channel}], dim=1)'
            node.value=ast.parse(f'self._compose_visible(fast_{channel},cold_{channel},{channel})',mode='eval').body
    env=TieredKVPolicy._visible_tuple.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree),'<controller-original-visibility>','exec'),env)
    return env['_visible_tuple']

VISIBLE=visibility_method()


class ControllerPrepare(BulkPrepare):
    def _prepare_decode_reads(self):
        if not self.native_prepare():
            self.controller_fallback_plans+=1
            return super()._prepare_decode_reads()


class NativeControllerPolicy(DeferredOwnershipPolicy,ControllerPrepare):
    direct_pool_cls=ControllerPool
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls=partial(ControllerCache,extra_slots=self.reserved_promotion_budget+self.max_decode_append)
        self.controller_plan_calls=self.controller_fallback_plans=0
        self.controller_rank_groups=self.controller_rank_h2d_bytes=0
        self.controller_cpu_streak_array_peak=0
        self.controller_selections={}
    def native_prepare(self):
        frames=[];layers=list(self.layer_caches)
        for layer,c in self.layer_caches.items():
            m=len(c.stt_pos)
            if any(t.device.type!='cpu' or not t.is_contiguous() or t.shape!=(m,) or t.dtype!=dtype
                    for t,dtype in [(c.stt_shadow,torch.bool),(c.stt_was_vram_resident,torch.bool),(c.stt_peek_streak,torch.float32)]):return False
            attention=(self._stt_attn or {}).get(layer)
            indexed=(self._stt_attn_idx or {}).get(layer)
            pairs=[]
            if attention:
                if indexed:pairs=[(value,i) for i,value in indexed.items() if i<m]
                else:
                    lookup={p:i for i,p in enumerate(c.stt_pos)}
                    pairs=[(value,lookup[p]) for p,value in attention.items() if p in lookup]
            if any(i<0 or math.isnan(value) for value,i in pairs):return False
            frames.append(dict(shadow=c.stt_shadow.numpy(),resident=c.stt_was_vram_resident.numpy(),streak=c.stt_peek_streak.numpy(),attention=pairs,floor=c.cfg.stt_live_floor,inclusive=c.cfg.inclusive))
        hysteresis={c._HYSTERESIS_MIN_STREAK for c in self.layer_caches.values()}
        if len(hysteresis)!=1:return False
        result=kernel().promotion(frames,self.promotion_budget,next(iter(hysteresis)))
        self._planned_promotions={};selections={};self.controller_plan_calls+=1
        self.controller_cpu_streak_array_peak=max(self.controller_cpu_streak_array_peak,sum(v['streak'].nbytes for v in result))
        for layer,value in zip(layers,result):
            c=self.layer_caches[layer];c.stt_peek_streak=torch.from_numpy(value['streak'])
            self._planned_promotions[layer]=(value['hits'],value['deferred'])
            if value['hits']:
                c.controller_rows=(value['hits'],value['rows'],value['floor_deferred'])
                if value['rows']:selections[layer]=value['rows']
        self._prefetch(selections)
        return True
    def controller_live(self,past):
        live={i:np.flatnonzero(~self.layer_caches[i].stt_shadow.numpy()).tolist()
              for i in range(len(past)) if i in self.layer_caches}
        if not live:return live
        count=min(self.stt_expose_quota,min(map(len,live.values())))
        groups={};self.controller_selections={}
        for layer,indices in live.items():
            c=self.layer_caches[layer]
            if count>=len(indices):self.controller_selections[id(c)]=indices;continue
            stamp=c.stt_last_access
            if stamp.device.type!='cuda' or stamp.dtype!=torch.float32 or not stamp.is_contiguous():continue
            groups.setdefault((len(indices),stamp.device),[]).append((layer,c,indices))
        for (rows,device),group in groups.items():
            states=[]
            for _,c,indices in group:
                stamp=c.stt_last_access;bits=stamp.view(torch.bfloat16).reshape(1,-1,2)
                states.append(SimpleNamespace(sram_k=bits,sram_v=bits,sram_cum=stamp,sram_age=stamp))
            gather(states,[indices for _,_,indices in group])
            # Native gather creates one shared [2,layers,1,rows,2] owner.
            rank=torch.stack([s.sram_k.reshape(rows,2).view(torch.float32).reshape(rows) for s in states])
            order=torch.argsort(rank,dim=1,descending=True)[:,:count].tolist()
            for (_,c,indices),selected in zip(group,order):self.controller_selections[id(c)]=sorted(indices[i] for i in selected)
            self.controller_rank_groups+=1
            self.controller_rank_h2d_bytes+=len(group)*(4*8+rows*4+4)
        return live
    def _visible_cold_indices(self,cache,live,count,bool_live_gather):
        if id(cache) in self.controller_selections:return self.controller_selections.pop(id(cache))
        return TieredKVPolicy._visible_cold_indices(cache,live,count,bool_live_gather)
    def _visible_tuple(self,past):
        # Preserve bounded gather phase; dispatch the generated original visible
        # function through this instance, never alter a module-global callback.
        return CONTROLLER_BOUNDARY(self,past)
    def reset(self):
        super().reset();self.controller_plan_calls=self.controller_fallback_plans=0
        self.controller_rank_groups=self.controller_rank_h2d_bytes=0
        self.controller_cpu_streak_array_peak=0;self.controller_selections={}


def controller_boundary():
    # Execute DeferredOwnershipPolicy's bounded phase, then preserve the raw
    # wrapper (clears consumed buffers) and reuse phase. Replace only final
    # visibility at instance scope using a separate generated method.
    source=inspect.getsource(ReuseVisiblePolicy._visible_tuple)
    tree=ast.parse(textwrap.dedent(source))
    last=tree.body[0].body[-1];assert ast.unparse(last)=='return VISIBLE_REUSE(self, past)'
    last.value.func=ast.Name(id='VISIBLE',ctx=ast.Load())
    env=ReuseVisiblePolicy._visible_tuple.__globals__.copy();env['VISIBLE']=VISIBLE
    exec(compile(ast.fix_missing_locations(tree),'<controller-reuse-boundary>','exec'),env)
    reuse=env['_visible_tuple']
    # Deferred/Raw methods currently dispatch super(); a standalone wrapper
    # runs their same gather and cleanup sequence without process-wide patching.
    from bounded_deferred_ownership import GROUP_LAYERS,bounded_gather,gather_contract
    from raw_input_reuse import gather_in_place
    from reserved_decode_append import reserve_visible
    def boundary(self,past):
        groups={}
        for c in self.layer_caches.values():
            if c.pending_keep is not None and c.sram_k.device.type=='cuda' and gather_contract(c):
                groups.setdefault((c.sram_k.shape[0],c.sram_k.shape[2],len(c.pending_keep),c.sram_k.device),[]).append(c)
        try:
            for group in groups.values():
                for start in range(0,len(group),GROUP_LAYERS):
                    size=bounded_gather(group[start:start+GROUP_LAYERS]);self.bounded_scratch_peak_bytes=max(size,self.bounded_scratch_peak_bytes);self.bounded_native_groups+=1
            for c in self.layer_caches.values():
                if c.pending_keep is not None and c._hot_buffer is not None:
                    gather_in_place(c);c.pending_keep=None;c.reused_compactions=getattr(c,'reused_compactions',0)+1;self.raw_compactions+=1;self.reused_layers+=1
            return reserve_visible(self,reuse(self,past))
        finally:
            self.controller_selections={}
            for c in self.layer_caches.values():c._hot_buffer=None
    return boundary
CONTROLLER_BOUNDARY=controller_boundary()
