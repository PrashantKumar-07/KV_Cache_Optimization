"""Frozen caller transform: native last-token logits on prefill only."""
import ast
from pathlib import Path

BASE = Path(__file__).resolve().parent


def compile_caller(filename, function_name, namespace, candidate):
    source = (BASE / (filename + '.control')).read_text()
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == function_name]
    assert len(functions) == 1
    function = functions[0]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'model'
        and not any(keyword.arg == 'output_attentions' for keyword in node.keywords)]
    assert len(calls) == (2 if function_name == 'generate_with_policy' else 1)
    original = ast.dump(function)
    if candidate:
        for call in calls:
            assert not any(keyword.arg == 'logits_to_keep' for keyword in call.keywords)
            call.keywords.append(ast.keyword(arg='logits_to_keep', value=ast.Constant(value=1)))
        changed = ast.dump(function)
        for call in calls:
            keyword = call.keywords.pop()
            assert keyword.arg == 'logits_to_keep' and keyword.value.value == 1
        assert ast.dump(function) == original
        for call in calls:
            call.keywords.append(ast.keyword(arg='logits_to_keep', value=ast.Constant(value=1)))
        assert ast.dump(function) == changed
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    environment = namespace.copy()
    exec(compile(module, str(BASE / filename), 'exec'), environment)
    return environment[function_name]
