"""Opt-in composition of verified exact bulk CPU promotion planning."""
import ast
import hashlib
import inspect
import textwrap
from functools import partial
import bulk_promotion_stage as stage
from cpu_batched_cache import CpuBatchPolicy
from fused_bounded_pool import FusedBoundedCache
from native_reuse_visible import NativeReusePolicy


def prepare_method():
    method = CpuBatchPolicy._prepare_decode_reads
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    before = ast.dump(tree,include_attributes=False)
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
        isinstance(n.func,ast.Attribute) and n.func.attr == '_promotion_plan']
    assert len(calls) == 1
    node = calls[0]
    assert ast.unparse(node.func) == 'TieredKVPolicy._promotion_plan'
    original = node.func
    node.func = ast.Name(id='_bulk_plan',ctx=ast.Load())
    after = ast.dump(tree,include_attributes=False)
    node.func = original
    assert ast.dump(tree,include_attributes=False) == before
    node.func = ast.Name(id='_bulk_plan',ctx=ast.Load())
    namespace = method.__globals__.copy(); namespace['_bulk_plan'] = stage.plan
    exec(compile(ast.fix_missing_locations(tree),'<bulk-original-CPU-prepare>','exec'),namespace)
    return namespace[method.__name__],dict(control=hashlib.sha256(before.encode()).hexdigest(),
        candidate=hashlib.sha256(after.encode()).hexdigest())


BULK_PREPARE, PREPARE_AST = prepare_method()


class BulkCache(FusedBoundedCache):
    resolve_promotion_indices = stage.resolve
    promotion_rows = stage.rows


class BulkPrepare(CpuBatchPolicy):
    _prepare_decode_reads = BULK_PREPARE


class BulkPromotionPolicy(NativeReusePolicy,BulkPrepare):
    def __init__(self,**options):
        super().__init__(**options)
        self.direct_cache_cls = partial(BulkCache,
            extra_slots=self.reserved_promotion_budget+self.max_decode_append)
