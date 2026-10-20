"""Explicit upstream BF16 index-layout correction; original source stays intact.

The indexed CPU keys are head-major. Project the same head-major keys rather
than reshaping contiguous sequence-major keys into unrelated head groups.
This is not a new cache policy. It must be disclosed and GPU/DCI validated.
"""
import ast
from contextlib import contextmanager
import hashlib,inspect,textwrap

_active=False

def prepare(module):
 original=module._icecache_prefill
 source=textwrap.dedent(inspect.getsource(original))
 tree=ast.parse(source);before=ast.dump(tree,include_attributes=False)
 nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='key']
 assert len(nodes)==1
 node=nodes[0]
 expected='key_states.reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]'
 assert ast.unparse(node.value)==expected,'Upstream projection changed; re-review needed'
 old=node.value
 node.value=ast.parse('key_states.transpose(1, 2).reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]',mode='eval').body
 modified=ast.dump(tree,include_attributes=False)
 node.value=old;assert ast.dump(tree,include_attributes=False)==before
 node.value=ast.parse('key_states.transpose(1, 2).reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]',mode='eval').body
 namespace=original.__globals__.copy()
 exec(compile(ast.fix_missing_locations(tree),'<IceCache-BF16-head-layout-correction>','exec'),namespace)
 return namespace['_icecache_prefill'],{'original_function_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'original_AST_sha256':hashlib.sha256(before.encode()).hexdigest(),'corrected_AST_sha256':hashlib.sha256(modified.encode()).hexdigest(),'only_change':'Transpose sequence/head axes before the existing reshape/projection; batch1 upstream supported path; allother functionstatements andcache/index algorithms unchanged.'}

@contextmanager
def corrected_projection():
 global _active
 if _active:raise RuntimeError('Nested projection correction is unsupported')
 from icecache.adapter import modeling
 original=modeling._icecache_prefill
 corrected,identity=prepare(modeling)
 _active=True
 modeling._icecache_prefill=corrected
 try:yield identity
 finally:
  modeling._icecache_prefill=original
  _active=False

def cpu_check():
 import torch
 from icecache.adapter import modeling
 original=modeling._icecache_prefill
 _,identity=prepare(modeling)
 cases=[]
 for heads in (1,2,8):
  n,d=256,128
  k=torch.zeros(1,n,heads,d,dtype=torch.bfloat16)
  k[0,:,:,0]=torch.arange(n)[:,None];k[0,:,:,1]=torch.arange(heads)[None,:]
  reference=torch.stack([k[0,:,h] for h in range(heads)])[:,32:224]
  corrected=k.transpose(1,2).reshape(heads,-1,d)[:,32:224]
  raw=k.reshape(heads,-1,d)[:,32:224]
  assert torch.equal(corrected,reference)
  assert torch.equal(raw,reference)==(heads==1)
  cases.append({'KVheads':heads,'corrected_exact':True,'original_exact':heads==1})
 with corrected_projection():
  assert modeling._icecache_prefill is not original
  try:
   with corrected_projection():pass
  except RuntimeError:pass
  else:raise AssertionError('Nested correction accepted')
 assert modeling._icecache_prefill is original
 try:
  with corrected_projection():raise ValueError('Injected failure')
 except ValueError:pass
 assert modeling._icecache_prefill is original
 return {'CPU_verified':True,'cases':cases,'AST_identity':identity,'restoration_normal_and_exception':True,'GPU_DCI_verified':False,'not_a_comparative_result':True}

if __name__=='__main__':
 from pathlib import Path
 import json
 out=Path(__file__).with_name('projection_correction_cpu_proof.json');assert not out.exists()
 result=cpu_check();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
