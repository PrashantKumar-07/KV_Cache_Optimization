"""Research query composition: retain visible hot prefix, replace cold suffix."""
import ast
import hashlib
import inspect
import textwrap

import torch
from transformers.models.mistral.modeling_mistral import rotate_half
from gpu_query_read_indices import GpuReadQuerySelector, READ_AST
from pool_query_selector import PoolQuerySelector


def replace_cold_suffix(layer, fast_rows, cold_key, cold_value, relative):
    """Indices come from unchanged topk/searchsorted; prepared prefix is trusted.

    Reject geometry/ownership mismatches before writing either destination.
    Caller must provide the unmodified ordinary prepared visible hot prefix.
    """
    key, value = layer.keys, layer.values
    if (key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1
            or cold_key.ndim != 3 or cold_key.shape != cold_value.shape
            or key.shape[1] != cold_key.shape[0] or key.shape[3] != cold_key.shape[2]
            or relative.ndim != 1 or relative.dtype != torch.int64 or fast_rows < 0
            or key.shape[2] != fast_rows + len(relative)
            or any(t.dtype != key.dtype or t.device != key.device
                   for t in (value, cold_key, cold_value))
            or relative.device != key.device or any(t.requires_grad for t in (key, value, cold_key, cold_value))):
        raise ValueError('Visible suffix geometry/dtype/device differs from prepared cache')
    if any(t.stride(3) != 1 or t.stride(2) < t.shape[3]
           or t.stride(1) < t.shape[2] * t.stride(2) for t in (key, value)):
        raise ValueError('Visible suffix requires nonoverlapping row/head strides')
    outputs = {t.untyped_storage()._cdata for t in (key, value)}
    sources = {t.untyped_storage()._cdata for t in (cold_key, cold_value, relative)}
    if len(outputs) != 2 or outputs & sources:
        raise ValueError('Visible suffix source/destination ownership overlaps')
    torch.index_select(cold_key, 1, relative, out=key[0, :, fast_rows:, :])
    torch.index_select(cold_value, 1, relative, out=value[0, :, fast_rows:, :])


def suffix_factory():
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
    control = ast.dump(tree, include_attributes=False)
    assert hashlib.sha256(control.encode()).hexdigest() == READ_AST['candidate']
    select = tree.body[0].body[0]
    start = next(i for i, node in enumerate(select.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'keys')
    block = select.body[start:start+5]
    expected = ast.parse('''keys=torch.cat((cache.sram_k[:,keep,:],cold_key.index_select(1,relative)),dim=1)
values=torch.cat((cache.sram_v[:,keep,:],cold_value.index_select(1,relative)),dim=1)
if keys.shape[-2]!=layer.keys.shape[-2]:
    raise RuntimeError('Query selection changed shared causal-mask length')
layer.keys=keys.unsqueeze(0).to(layer.keys.dtype)
layer.values=values.unsqueeze(0).to(layer.values.dtype)
''').body
    assert ast.dump(ast.Module(body=block, type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=expected, type_ignores=[]), include_attributes=False)
    replacement = ast.parse('''replace_cold_suffix(layer,len(keep),cold_key,cold_value,relative)
self.visible_suffix_updates+=1
''').body
    select.body[start:start+5] = replacement
    candidate = ast.dump(tree, include_attributes=False)
    select.body[start:start+2] = block
    assert ast.dump(tree, include_attributes=False) == control
    select.body[start:start+5] = replacement
    namespace = dict(torch=torch, rotate_half=rotate_half, replace_cold_suffix=replace_cold_suffix)
    exec(compile(ast.fix_missing_locations(tree), '<query-visible-suffix>', 'exec'), namespace)
    return namespace['_hook'], dict(control=hashlib.sha256(control.encode()).hexdigest(),
        candidate=hashlib.sha256(candidate.encode()).hexdigest(), only_composition_block_changed=True)


SUFFIX_HOOK, SUFFIX_AST = suffix_factory()


class SuffixQuerySelector(GpuReadQuerySelector):
    _hook = SUFFIX_HOOK

    def __init__(self, model, policy):
        self.visible_suffix_updates = 0
        super().__init__(model, policy)
