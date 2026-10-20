"""Read-only isolated CPU ownership edge reproductions; no CUDA calls."""
from pathlib import Path
import hashlib,json,sys,threading
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import reusable_exact_root as m
from cold_boundary_policy import ColdBoundaryPolicy

barrier=threading.Barrier(2);results=[];errors=[];allocations=[]
def key(device):return threading.get_ident(),0,99
def allocate(shape,**kw):
 allocations.append(threading.get_ident());barrier.wait(timeout=5)
 return torch.zeros(shape,dtype=kw['dtype'])
owner=m.ExactRootOwner()
def worker():
 try:results.append(owner.acquire((2,4,1,2),dtype=torch.bfloat16,device='cuda:0'))
 except BaseException as error:errors.append(repr(error))
with patch.object(m,'context_key',key),patch.object(m,'allocate',allocate):
 threads=[threading.Thread(target=worker) for _ in range(2)]
 for t in threads:t.start()
 for t in threads:t.join(timeout=8)
 assert all(not t.is_alive() for t in threads)
 print({"debug_results":len(results),"debug_errors":errors,"debug_leases":owner.leases,"debug_allocations":allocations})
 assert len(results)==2 and not errors and owner.leases==2
 concurrent={'simultaneous_initial_acquires_succeeded':len(results),'allocations':len(allocations),'leases':owner.leases,'intended_rejection_failed':True}

with patch.object(m,'context_key',lambda device:(1,0,99)),patch.object(m,'allocate',lambda shape,**kw:torch.zeros(shape,dtype=kw['dtype'])):
 a,b=m.ExactRootOwner(),m.ExactRootOwner()
 for o in [a,b]:o.acquire((2,4,1,2),dtype=torch.bfloat16,device='cuda:0')
 tokena,tokenb=a.token,b.token
 policy=m.ReusableExactPolicy(root_owner=a)
 manager=m.ReusableBulkPool.__new__(m.ReusableBulkPool);manager.root_owner=b;manager.root_token=tokenb
 cache=SimpleNamespace(pool=b.root,manager=manager)
 cache.close=lambda:None
 manager.caches={0:cache};policy.manager=manager;policy.layer_caches={0:cache};cleanup=[]
 def parent_reset(p):
  cleanup.append('foreign_manager_cleanup_executed');cache.pool=None;cache.manager=None;p.manager=None;p.layer_caches={}
 with patch.object(ColdBoundaryPolicy,'reset',parent_reset):
  try:policy.reset()
  except RuntimeError as error:assert 'Foreign' in str(error)
  else:raise AssertionError('Foreigntoken should reject')
 assert cleanup and cache.pool is None and a.token is tokena and b.token is tokenb
 foreign={'parent_cleanup_occurred_before_foreign_owner_rejection':True,'both_owner_tokens_still_active':True}
 for o in [a,b]:o.release(o.token);o.close()
out={'CPU_only':True,'source_sha256':hashlib.sha256((ROOT/'experiments/reusable_exact_root.py').read_bytes()).hexdigest(),
 'concurrent_first_acquire':concurrent,'foreign_manager':foreign,
 'scope':'Contract/failure probes using mocked contexts/allocation/reset. Not native CUDA lifetime proof. Intended serialized sameowner usage unaffected.',
 'minimal_remedies':['Reserve provisioning busy state before context/allocation and reject close/acquire while provisioning; rollback reservation on failure',
 'Validate manager.root_owner is self.root_owner and active token before mutating/resetting any manager/cache']}
print(json.dumps(out,indent=2))
