"""Whole-body preservation proof for extracting the production victim selector."""
import ast
import copy
from pathlib import Path


def test_inlined_selector_restores_entire_original_eviction_body():
    root=Path(__file__).resolve().parents[1]
    old=ast.parse((root/'results/hardware/batched_eviction_coordinator_20261001/tiered_kv_cache_before.py').read_text())
    current=ast.parse((root/'src/tiered_kv_cache.py').read_text())
    methods=lambda tree:{n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    previous,now=methods(old),methods(current)
    selection=now['_select_hot_victims']
    # Exclude its docstring/cfg alias and return; evict_and_demote already owns cfg.
    body=copy.deepcopy(selection.body[2:-1])
    expanded=copy.deepcopy(now['evict_and_demote']);calls=0
    for i,node in enumerate(expanded.body):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_select_hot_victims':
            expanded.body[i:i+1]=body;calls+=1;break
    assert calls==1
    assert ast.dump(expanded,include_attributes=False)==ast.dump(previous['evict_and_demote'],include_attributes=False)
