"""Research cross-layer victim projections; original selection math/fallback."""
import ast
import hashlib
import inspect
import textwrap
import torch
from torch.nn.utils.rnn import pad_sequence
from batched_eviction_policy import BatchedEvictionPolicy
from probe_batched_eviction_selection import batched_victims
from bulk_promotion_policy import BulkPromotionPolicy


def projected_scores(caches,n_new):
    cumulative=torch.nn.functional.pad(pad_sequence([c.sram_cum for c in caches],batch_first=True),(0,n_new))
    age=torch.nn.functional.pad(pad_sequence([c.sram_age for c in caches],batch_first=True),(0,n_new))
    lengths=torch.tensor([c.sram_age.numel() for c in caches],device=age.device)
    existing=torch.arange(age.shape[1],device=age.device)[None,:]<lengths[:,None]
    age=torch.where(existing,age+1,0)
    return cumulative,age


def selection_factory():
    tree=ast.parse(textwrap.dedent(inspect.getsource(batched_victims)))
    before=ast.dump(tree,include_attributes=False);body=tree.body[0].body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='cumulative')
    assert ast.unparse(body[start].value)=='pad_sequence([c.sram_cum for c in chosen], batch_first=True)'
    assert ast.unparse(body[start+1].value)=='pad_sequence([c.sram_age for c in chosen], batch_first=True)'
    body[start:start+2]=ast.parse('cumulative,age=projected_scores(chosen,n_new)').body
    tree.body[0].args.args.append(ast.arg(arg='n_new'));tree.body[0].args.defaults.append(ast.Constant(value=1))
    namespace=batched_victims.__globals__.copy();namespace['projected_scores']=projected_scores
    exec(compile(ast.fix_missing_locations(tree),'<batched-projected-victims>','exec'),namespace)
    return namespace['batched_victims'],dict(control=hashlib.sha256(before.encode()).hexdigest(),candidate=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())


PROJECTED_VICTIMS,SELECTION_AST=selection_factory()


def coordinator_factory():
    method=BatchedEvictionPolicy._prepare_decode_reads
    tree=ast.parse(textwrap.dedent(inspect.getsource(method)));before=ast.dump(tree,include_attributes=False)
    changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='zeros':
            assert ast.unparse(node.value)=='torch.zeros(n_new, device=cache.cfg.device, dtype=cache.cfg.dtype)'
            node.value=ast.Constant(value=None);changed.append('unusedzeros')
        if isinstance(node,ast.keyword) and node.arg in ('sram_cum','sram_age'):
            expected={'sram_cum':'torch.cat((cache.sram_cum, zeros))','sram_age':'torch.cat((cache.sram_age + 1, zeros))'}
            assert ast.unparse(node.value)==expected[node.arg]
            node.value=ast.parse('cache.'+node.arg,mode='eval').body;changed.append(node.arg)
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='batched_victims':
            node.keywords.append(ast.keyword(arg='n_new',value=ast.Name(id='n_new',ctx=ast.Load())));changed.append('selection')
    assert sorted(changed)==['selection','sram_age','sram_cum','unusedzeros']
    namespace=method.__globals__.copy();namespace['batched_victims']=PROJECTED_VICTIMS
    exec(compile(ast.fix_missing_locations(tree),'<batched-projection-coordinator>','exec'),namespace)
    return namespace['_prepare_decode_reads'],dict(control=hashlib.sha256(before.encode()).hexdigest(),candidate=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),changes=changed)


PROJECTED_PREPARE,COORDINATOR_AST=coordinator_factory()
class ProjectedEvictionPolicy(BulkPromotionPolicy):
    _prepare_decode_reads=PROJECTED_PREPARE
