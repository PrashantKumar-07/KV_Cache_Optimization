"""Prove deferred eviction changes only the production compaction hook."""
import ast
import copy
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments')]
from fused_eviction_policy import CONTROL_AST,DEFERRED_AST,eviction_method


def test_deferred_eviction_preserves_entire_noncompaction_body():
    # Build independent ASTs from the method factory, then restore one body.
    import inspect,textwrap
    from tiered_kv_cache import TieredKVCache
    old=ast.parse(textwrap.dedent(inspect.getsource(TieredKVCache.evict_and_demote)))
    new=copy.deepcopy(old)
    node=next(n for n in new.body[0].body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='victims')
    node.body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),
        attr='_compact_or_queue',ctx=ast.Load()),args=[ast.Name(id='victims',ctx=ast.Load())],keywords=[]))]
    assert ast.dump(old)==CONTROL_AST
    assert ast.dump(new)==DEFERRED_AST
    function,compact,original,modified=eviction_method()
    assert original==CONTROL_AST and modified==DEFERRED_AST
    assert function.__code__.co_filename=='<deferred_production_eviction>'
    assert compact.__code__.co_filename=='<production_eager_compaction>'
