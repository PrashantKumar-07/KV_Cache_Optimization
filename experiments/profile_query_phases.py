"""Diagnostic ranges around the unchanged GPU-index full-query selector."""
import ast
import hashlib
import inspect
import textwrap

import torch
from transformers.models.mistral.modeling_mistral import rotate_half
from gpu_query_read_indices import GpuReadQuerySelector, READ_AST
from pool_query_selector import PoolQuerySelector


PHASES = {
    'live': 'live_index_lookup',
    'cos': 'query_projection_rotary',
    'keep': 'fast_key_materialization',
    'cold_key': 'native_cold_pair_read',
    'fast_logits': 'attention_ranking',
    'chosen': 'topk_cpu_ids',
    'relative': 'gpu_visible_composition',
    'self.policy._vis_stt_pos[index]': 'cpu_visible_metadata',
}


def annotated_factory():
    tree = ast.parse(textwrap.dedent(inspect.getsource(PoolQuerySelector._hook)))
    changes = []

    class ReadIndices(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and ast.unparse(node.targets[0]) == 'live':
                assert ast.unparse(node.value) == 'torch.where(~cache.stt_shadow)[0].to(cache.cfg.device)'
                node.value = ast.parse('self.policy.query_reader.indices(index)', mode='eval').body
                changes.append('live')
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == 'cache.select_stt_pair':
                assert ast.unparse(node) == 'cache.select_stt_pair(live.tolist())'
                changes.append('paired_read')
                return ast.parse('self.policy.query_reader.read_live(index)', mode='eval').body
            return self.generic_visit(node)

    tree = ReadIndices().visit(tree)
    assert sorted(changes) == ['live', 'paired_read']
    candidate = ast.dump(tree, include_attributes=False)
    assert hashlib.sha256(candidate.encode()).hexdigest() == READ_AST['candidate']
    select = tree.body[0].body[0]
    assert isinstance(select, ast.FunctionDef) and select.name == 'select'
    body = []
    current = None
    observed = []
    for statement in select.body:
        if isinstance(statement, ast.Assign):
            target = statement.targets[0]
            first = target.elts[0] if isinstance(target, ast.Tuple) else target
            phase = PHASES.get(ast.unparse(first))
            if phase:
                observed.append(phase)
                current = ast.With(items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id='record_function', ctx=ast.Load()),
                    args=[ast.Constant(value='query_phase/' + phase)], keywords=[]))], body=[])
                body.append(current)
        if current is None:
            body.append(statement)
        else:
            current.body.append(statement)
    assert observed == list(PHASES.values())
    select.body = body
    annotated = ast.dump(tree, include_attributes=False)

    class Unwrap(ast.NodeTransformer):
        def visit_With(self, node):
            assert ast.unparse(node.items[0].context_expr).startswith('record_function(')
            return node.body

    tree = ast.fix_missing_locations(tree)
    restored = Unwrap().visit(ast.parse(ast.unparse(tree)))
    assert ast.dump(restored, include_attributes=False) == candidate
    namespace = dict(torch=torch, rotate_half=rotate_half,
                     record_function=torch.profiler.record_function)
    exec(compile(ast.fix_missing_locations(tree), '<query-phase-annotations>', 'exec'), namespace)
    return namespace['_hook'], {
        'candidate_ast_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
        'annotated_ast_sha256': hashlib.sha256(annotated.encode()).hexdigest(),
        'annotations_unwrap_to_exact_candidate': True,
        'phases': observed,
    }


PHASE_HOOK, PHASE_AST = annotated_factory()


class ProfiledGpuQuerySelector(GpuReadQuerySelector):
    _hook = PHASE_HOOK
