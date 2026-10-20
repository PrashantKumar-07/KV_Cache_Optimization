"""Research batched GPU read-index maps for the unchanged full-query selector."""
import ast,hashlib,inspect,textwrap
import numpy as np
import torch
from transformers.models.mistral.modeling_mistral import rotate_half
from bulk_promotion_policy import BulkPromotionPolicy
from pool_query_selector import PoolQuerySelector


def version(cache):
    return (cache.base,cache.capacity,tuple(cache._logical),tuple(cache.stt_pos),cache.stt_shadow.numpy().tobytes())


def source_maps(caches):
    layers=list(caches);assert layers==list(range(len(layers))) and layers,'Complete ordered layers required'
    first=caches[0];heads=first.cfg.num_heads;capacity=first.capacity
    if heads<1 or capacity<1 or len(layers)*heads*capacity>np.iinfo(np.int32).max:
        raise ValueError('Native source addresses exceed positive int32 geometry')
    blocks=np.zeros((len(layers),heads,capacity),dtype=np.int32)
    live=np.zeros((len(layers),capacity),dtype=np.int64);lengths=[];versions=[]
    for i,cache in caches.items():
        logical=np.asarray(cache._logical,dtype=np.int64);shadow=cache.stt_shadow.numpy()
        if (cache.capacity!=capacity or cache.cfg.num_heads!=heads or cache.base!=i*heads*capacity
                or len(logical)!=len(cache.stt_pos) or shadow.shape!=(len(logical),) or shadow.dtype!=np.bool_
                or len(logical)>capacity or np.any(logical<0) or np.any(logical>=capacity)
                or len(np.unique(logical))!=len(logical)):
            raise ValueError('Invalid native source-map geometry/physical slots')
        indices=np.flatnonzero(~shadow);count=len(indices);lengths.append(count);versions.append(version(cache))
        live[i,:count]=indices
        blocks[i,:,:count]=cache.base+np.arange(heads,dtype=np.int32)[:,None]*capacity+logical[indices][None,:]
    return blocks,live,lengths,versions


class GpuLiveReader:
    def __init__(self,policy):
        self.policy=policy;self.blocks=self.live=self.ready=None;self.lengths=self.versions=[]
        self.manager=None;self.metadata_h2d_bytes=self.refreshes=self.reads=0
    def refresh(self):
        blocks,live,self.lengths,self.versions=source_maps(self.policy.layer_caches)
        self.manager=self.policy.manager
        with torch.cuda.device(self.manager.device):
            self.blocks=torch.from_numpy(blocks).to(self.manager.device)
            self.live=torch.from_numpy(live).to(self.manager.device)
            self.ready=torch.cuda.Event();self.ready.record()
        self.metadata_h2d_bytes+=blocks.nbytes+live.nbytes;self.refreshes+=1
    def indices(self,layer):
        if self.manager is not self.policy.manager or self.versions[layer]!=version(self.policy.layer_caches[layer]):
            raise RuntimeError('GPU source-map snapshot is stale')
        torch.cuda.current_stream(self.manager.device).wait_event(self.ready)
        self.live.record_stream(torch.cuda.current_stream(self.manager.device))
        return self.live[layer,:self.lengths[layer]]
    def read_live(self,layer):
        self.indices(layer);cache=self.policy.layer_caches[layer];manager=self.manager
        if cache._prefetched is not None:raise RuntimeError('Unconsumed ordinary prefetched cold rows')
        manager.flush();heads=cache.cfg.num_heads;rows=self.lengths[layer];count=heads*rows
        with torch.cuda.device(manager.device):
            manager.wait();self.blocks.record_stream(torch.cuda.current_stream(manager.device))
            source=self.blocks[layer,:,:rows].contiguous().flatten()
            output=torch.empty((2,count,1,manager.dim),dtype=manager.pool.dtype,device=manager.device)
            if count:
                manager.h2d.register_cpu_kv_cache_pinned(manager.pool)
                destination=torch.arange(count,dtype=torch.int32,device=manager.device)
                manager.h2d.kv_cache_h2d_transfer(manager.pool,output,source,destination)
                manager.h2d_calls+=1;cache.h2d_calls+=1;manager.record()
        cache.host_read_bytes+=count*2*manager.dim*manager.pool.element_size();self.reads+=1
        pair=output.reshape(2,heads,rows,manager.dim).to(cache.cfg.resolved_kv_dtype)
        return pair[0],pair[1]
    def close(self):
        self.blocks=self.live=self.ready=self.manager=None;self.lengths=self.versions=[]
        self.metadata_h2d_bytes=self.refreshes=self.reads=0


class GpuReadQueryPolicy(BulkPromotionPolicy):
    def __init__(self,**options):
        self.query_reader=None;super().__init__(**options);self.query_reader=GpuLiveReader(self)
    def _visible_tuple(self,past):
        result=super()._visible_tuple(past);self.query_reader.refresh();return result
    def reset(self):
        if self.query_reader:self.query_reader.close()
        super().reset()


def selector_factory():
    tree=ast.parse(textwrap.dedent(inspect.getsource(PoolQuerySelector._hook)))
    before=ast.dump(tree,include_attributes=False);changes=[]
    class Replace(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and ast.unparse(node.targets[0])=='live':
                assert ast.unparse(node.value)=='torch.where(~cache.stt_shadow)[0].to(cache.cfg.device)'
                node.value=ast.parse('self.policy.query_reader.indices(index)',mode='eval').body;changes.append('live')
            return self.generic_visit(node)
        def visit_Call(self,node):
            if ast.unparse(node.func)=='cache.select_stt_pair':
                assert ast.unparse(node)=='cache.select_stt_pair(live.tolist())'
                changes.append('paired_read');return ast.parse('self.policy.query_reader.read_live(index)',mode='eval').body
            return self.generic_visit(node)
    candidate=Replace().visit(tree);assert sorted(changes)==['live','paired_read']
    namespace=dict(torch=torch,rotate_half=rotate_half)
    exec(compile(ast.fix_missing_locations(candidate),'<GPU-native-query-read-indices>','exec'),namespace)
    return namespace['_hook'],dict(control=hashlib.sha256(before.encode()).hexdigest(),candidate=hashlib.sha256(ast.dump(candidate,include_attributes=False).encode()).hexdigest())


GPU_READ_HOOK,READ_AST=selector_factory()
class GpuReadQuerySelector(PoolQuerySelector):_hook=GPU_READ_HOOK
