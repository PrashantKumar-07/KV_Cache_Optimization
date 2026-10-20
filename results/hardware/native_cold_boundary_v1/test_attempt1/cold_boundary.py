"""Research canonical cold boundary; grouped Torch ranking stays unchanged."""
from functools import lru_cache
import hashlib,os
from pathlib import Path
import numpy as np
import torch

@lru_cache(maxsize=1)
def kernel():
 from torch.utils.cpp_extension import load
 root=Path(__file__).parent;files=[root/'native_cold_boundary.cpp',root/'native_cold_boundary.cu'];digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
 build=Path('/data/nishant/tieredkv_external')/('cold_boundary_'+digest[:16]);build.mkdir(parents=True,exist_ok=True)
 os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
 return load(name='tieredkv_cold_boundary_v1',sources=list(map(str,files)),build_directory=str(build),extra_cflags=['-O2'],extra_cuda_cflags=['-O3','--fmad=false'])

def live_plan(owners,quota):
 if type(quota) is not int or quota<0:raise ValueError('Invalid exposure quota')
 fields=dict(cold=np.int64,shadow=np.bool_,slots=np.int32,cold_n=np.int32,hot_n=np.int32,version=np.int64)
 if any(k not in owners or not isinstance(owners[k],np.ndarray) or owners[k].dtype!=np.dtype(t) or not owners[k].flags.c_contiguous for k,t in fields.items()):raise ValueError('Invalid canonical CPU owners')
 shape=owners['cold'].shape
 if len(shape)!=2 or min(shape)<1 or any(owners[k].shape!=shape for k in ['shadow','slots']) or any(owners[k].shape!=(shape[0],) for k in ['cold_n','hot_n','version']):raise ValueError('Invalid CPU owner shape')
 if np.any(owners['cold_n']<0) or np.any(owners['cold_n']>shape[1]) or np.any(owners['hot_n']<1):raise ValueError('Invalid canonical counts')
 live=[np.flatnonzero(~owners['shadow'][i,:int(n)]).astype(np.int32) for i,n in enumerate(owners['cold_n'])]
 count=min(quota,min(map(len,live)))
 if count<1:raise ValueError('Empty exposed cold requires caller fallback')
 return live,count

def address_plan(owners,selected,bases,heads,pool_rows):
 if type(heads) is not int or heads<1 or type(pool_rows) is not int or pool_rows<1 or selected.ndim!=2 or selected.shape[0]!=len(owners['cold_n']) or selected.dtype!=np.int32 or bases.shape!=(selected.shape[0],) or bases.dtype!=np.int64:raise ValueError('Invalid physical address plan')
 if np.any(selected<0) or np.any(selected>=owners['cold_n'][:,None]) or np.any(np.diff(selected,axis=1)<=0):raise ValueError('Invalid selected cold indices')
 if np.any(np.take_along_axis(owners['shadow'],selected,axis=1)):raise ValueError('Selected shadow row')
 slots=np.take_along_axis(owners['slots'],selected,axis=1);capacity=owners['slots'].shape[1]
 if np.any(slots<0) or np.any(slots>=capacity):raise ValueError('Invalid physical slot')
 addresses=bases[:,None,None]+np.arange(heads,dtype=np.int64)[None,:,None]*capacity+slots[:,None,:]
 if np.any(addresses<0) or np.any(addresses>=pool_rows) or np.any(addresses>np.iinfo(np.int32).max) or len(np.unique(addresses))!=addresses.size:raise ValueError('Physical addresses overlap or exceed pool')
 return addresses.astype(np.int32).reshape(-1),np.take_along_axis(owners['cold'],selected,axis=1)

def visible(policy,past):
 """Mutate only reserved cold tails; retain CPU state and physical IO ordering."""
 cpu=policy.cpu_arena;arena=policy.state_arena;manager=policy.manager;caches=list(policy.layer_caches.values());layers=len(caches)
 if cpu is None or arena is None or manager is None or layers<1 or list(policy.layer_caches)!=list(range(layers)) or len(past)!=layers:raise ValueError('Canonical complete cold boundary required')
 owners=cpu.owners;live,count=live_plan(owners,policy.stt_expose_quota);versions=owners['version'].copy();hot=int(owners['hot_n'][0]);cold=arena.cold;device=cold.device
 if cold.ndim!=2 or cold.device.type!='cuda' or cold.dtype!=torch.float32 or not cold.is_contiguous() or cold.shape!=owners['cold'].shape or np.any(owners['hot_n']!=hot):raise ValueError('Uniform CUDA metadata boundary required')
 heads,_,dim=caches[0].sram_k.shape;roots=[];desc=np.empty((layers,4),np.int64);bases=np.asarray([c.base for c in caches],np.int64)
 if manager.pool is None or manager.pool.dtype!=torch.bfloat16 or manager.pool.device.type!='cpu' or not manager.pool.is_pinned() or manager.dim!=dim or manager.geometry[:2]!=(heads,owners['cold'].shape[1]):raise ValueError('Invalid mapped pool geometry')
 for i,c in enumerate(caches):
  if manager.caches.get(i) is not c or c.manager is not manager or c._cpu_arena is not cpu or c._state_arena is not arena or c._cpu_commands is not None or c.pending_keep is not None or c._prefetched is not None:raise ValueError('Unconsumed or foreign cold boundary')
  if c.stt_last_access.data_ptr()!=cold[i].data_ptr() or len(c.stt_last_access)!=int(owners['cold_n'][i]):raise ValueError('Cold timestamp owner mismatch')
  if len(past[i])!=2:raise ValueError('Invalid raw channels')
  pair=[c.sram_k,c.sram_v]
  for t,raw in zip(pair,past[i]):
   if t.ndim!=3 or t.shape!=(heads,hot,dim) or t.device!=device or t.dtype!=torch.bfloat16 or t.requires_grad or t.stride(2)!=1 or t.stride(1)!=dim or t.stride(0)%dim or t.stride(0)<(hot+count+1)*dim:raise ValueError('Unsupported hot root geometry')
   if raw.ndim!=4 or raw.shape[0]!=1 or raw.shape[1]!=heads or raw.shape[3]!=dim or raw.shape[2]<hot+count or raw.device!=device or raw.dtype!=t.dtype or raw.data_ptr()!=t.data_ptr() or raw.stride()[1:]!=t.stride() or raw.untyped_storage()._cdata!=t.untyped_storage()._cdata:raise ValueError('Raw payload is not the reserved hot root')
   if t.storage_offset()+(heads-1)*t.stride(0)+(hot+count+1)*dim>t.untyped_storage().nbytes()//2:raise ValueError('Reserved root extent exceeds storage')
   roots.append(t.untyped_storage()._cdata)
  if pair[0].stride()!=pair[1].stride():raise ValueError('Unequal channel pitches')
  desc[i]=[t.data_ptr() for t in pair]+[pair[0].stride(0)//dim,hot]
 if len(set(roots))!=len(roots) or cold.untyped_storage()._cdata in roots:raise ValueError('Aliased payload or metadata roots')
 # Validate all physical slots/bases before any GPU or IO operation. Actual
 # selected rows are validated again after the unchanged grouped ranking.
 if np.any(bases<0) or np.any(bases+heads*owners['cold'].shape[1]>manager.pool.shape[1]):raise ValueError('Pool slice exceeds storage')
 for i,n in enumerate(owners['cold_n']):
  slots=owners['slots'][i,:int(n)]
  if np.any(slots<0) or np.any(slots>=owners['cold'].shape[1]):raise ValueError('Invalid canonical slot')
 selected=np.empty((layers,count),np.int32);groups={};upload=desc.nbytes;rank_peak=0;cpu_bytes=sum(a.nbytes for a in live)+selected.nbytes+desc.nbytes+bases.nbytes+versions.nbytes
 for i,row in enumerate(live):
  if len(row)==count:selected[i]=row
  else:groups.setdefault(len(row),[]).append(i)
 stream=torch.cuda.current_stream(device);cold.record_stream(stream)
 for width,ids in groups.items():
  index=np.stack([live[i] for i in ids]);which=np.asarray(ids,np.int32);indices=torch.from_numpy(index).to(device);layer_ids=torch.from_numpy(which).to(device);rank=torch.empty(index.shape,device=device,dtype=torch.float32)
  kernel().rank(cold,indices,layer_ids,rank);order=torch.argsort(rank,dim=1,descending=True)[:,:count].cpu().numpy()
  selected[ids]=np.sort(np.take_along_axis(index,order,axis=1),axis=1)
  upload+=index.nbytes+which.nbytes;rank_peak=max(rank_peak,indices.numel()*4+layer_ids.numel()*4+rank.numel()*4);cpu_bytes=max(cpu_bytes,sum(a.nbytes for a in live)+selected.nbytes+desc.nbytes+bases.nbytes+versions.nbytes+index.nbytes+which.nbytes+order.nbytes)
  del indices,layer_ids,rank,order,index,which
 addresses,positions=address_plan(owners,selected,bases,heads,int(manager.pool.shape[1]));upload+=addresses.nbytes
 if not np.array_equal(versions,owners['version']):raise RuntimeError('Canonical CPU state changed during cold planning')
 source=torch.from_numpy(addresses).to(device);descriptors=torch.from_numpy(desc).to(device);rows=torch.empty((2,layers*heads*count,1,dim),dtype=torch.bfloat16,device=device);destination=torch.arange(rows.shape[1],dtype=torch.int32,device=device)
 manager.flush();manager.wait();manager.h2d.register_cpu_kv_cache_pinned(manager.pool);manager.h2d.kv_cache_h2d_transfer(manager.pool,rows,source,destination);manager.h2d_calls+=1;manager.record()
 for c in caches:
  c.host_read_bytes+=2*heads*count*dim*2;c.h2d_calls+=1
  c.sram_k.record_stream(stream);c.sram_v.record_stream(stream)
 rows.record_stream(stream);kernel().scatter(descriptors,rows,layers,heads,count)
 manager.controller_address_calls+=1;manager.controller_address_array_peak=max(manager.controller_address_array_peak,addresses.nbytes)
 policy.controller_rank_groups+=len(groups);policy.controller_rank_h2d_bytes+=upload-addresses.nbytes-desc.nbytes
 policy._stt_exposed_total+=count;policy._stt_exposure_samples+=1
 policy._vis_sram_keep={i:list(range(hot)) for i in range(layers)};policy._vis_stt_pos={i:positions[i].tolist() for i in range(layers)};policy._vis_stt_idx={i:selected[i].tolist() for i in range(layers)};policy._vis_len={i:hot+count for i in range(layers)}
 policy.shared_hot_prefixes+=layers;policy.controller_selections={}
 for c in caches:c._hot_buffer=None
 output=tuple(tuple(raw[:,:,:hot+count] for raw in pair) for pair in past)
 policy.cold_boundary_steps=getattr(policy,'cold_boundary_steps',0)+1
 policy.cold_boundary_descriptor_h2d_bytes=getattr(policy,'cold_boundary_descriptor_h2d_bytes',0)+upload
 policy.cold_boundary_rank_peak_bytes=max(getattr(policy,'cold_boundary_rank_peak_bytes',0),rank_peak)
 policy.cold_boundary_cpu_peak_bytes=max(getattr(policy,'cold_boundary_cpu_peak_bytes',0),cpu_bytes+addresses.nbytes+positions.nbytes)
 return output
