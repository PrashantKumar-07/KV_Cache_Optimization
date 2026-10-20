"""Research-only packing of the original post-promotion metadata projection."""
import ast
import hashlib
import inspect
import textwrap
from types import SimpleNamespace
import torch
import probe_batched_eviction_selection as selection
from batched_eviction_policy import BatchedEvictionPolicy


def reference_projection(rows, n_new):
    result = []
    for cache, positions in rows:
        zeros = torch.zeros(n_new, device=cache.cfg.device, dtype=cache.cfg.dtype)
        result.append(SimpleNamespace(cfg=cache.cfg, sram_pos=positions,
            sram_cum=torch.cat((cache.sram_cum, zeros)),
            sram_age=torch.cat((cache.sram_age + 1, zeros)),
            sram_grace=torch.cat((cache.sram_grace, torch.zeros(n_new, dtype=torch.bool)))))
    return result


def packed_projection(rows, n_new):
    if not rows: return []
    caches = [cache for cache, _ in rows]
    first = caches[0]
    if not all(c.cfg.dtype == torch.float32 and c.cfg.device == first.cfg.device
               and c.sram_cum.ndim == c.sram_age.ndim == 1
               and c.sram_cum.shape == c.sram_age.shape == first.sram_cum.shape
               and c.sram_cum.dtype == c.sram_age.dtype == torch.float32
               and c.sram_cum.device == c.sram_age.device == first.sram_cum.device
               and c.sram_cum.device.type == 'cuda' for c in caches):
        return reference_projection(rows, n_new)
    zeros = torch.zeros((len(caches), n_new), device=first.cfg.device, dtype=first.cfg.dtype)
    cumulative = torch.cat((torch.stack([c.sram_cum for c in caches]), zeros), dim=1)
    age = torch.cat((torch.stack([c.sram_age for c in caches]) + 1, zeros), dim=1)
    return [SimpleNamespace(cfg=c.cfg, sram_pos=positions,
                sram_cum=cumulative[i], sram_age=age[i],
                sram_grace=torch.cat((c.sram_grace, torch.zeros(n_new, dtype=torch.bool))))
            for i, (c, positions) in enumerate(rows)]


def stack_or_pad(sequences, *, batch_first, padding_value=0):
    if batch_first and sequences and all(t.shape == sequences[0].shape
            and t.dtype == sequences[0].dtype and t.device == sequences[0].device for t in sequences):
        return torch.stack(sequences)
    return selection.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)


def stacked_selector():
    tree = ast.parse(inspect.getsource(selection.batched_victims))
    original = ast.dump(tree, include_attributes=False)
    changed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'pad_sequence':
            node.func.id = 'stack_or_pad'; changed.append(node)
    assert len(changed) == 4
    modified = ast.dump(tree, include_attributes=False)
    for node in changed: node.func.id = 'pad_sequence'
    assert ast.dump(tree, include_attributes=False) == original
    for node in changed: node.func.id = 'stack_or_pad'
    namespace = vars(selection).copy(); namespace['stack_or_pad'] = stack_or_pad
    exec(compile(ast.fix_missing_locations(tree), '<stacked-original-selection>', 'exec'), namespace)
    return namespace['batched_victims'], original, modified


STACKED_VICTIMS, CONTROL_SELECTION_AST, STACKED_SELECTION_AST = stacked_selector()


def reference_identity():
    """The benchmark reference must keep the production projection statements."""
    production = ast.parse(textwrap.dedent(inspect.getsource(BatchedEvictionPolicy._prepare_decode_reads)))
    loop = next(n for n in production.body[0].body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
    reference = ast.parse(inspect.getsource(reference_projection))
    ref_loop = next(n for n in reference.body[0].body if isinstance(n, ast.For))
    tail = ast.Module(body=loop.body[-2:], type_ignores=[])
    for node in ast.walk(tail):
        if isinstance(node, ast.Name) and node.id == 'projections': node.id = 'result'
    expected = ast.dump(tail, include_attributes=False)
    actual = ast.dump(ast.Module(body=ref_loop.body, type_ignores=[]), include_attributes=False)
    assert actual == expected, 'Stage reference differs from production projection'
    return hashlib.sha256(expected.encode()).hexdigest()


def source_identity():
    return dict(control_selection_ast=hashlib.sha256(CONTROL_SELECTION_AST.encode()).hexdigest(),
                stacked_selection_ast=hashlib.sha256(STACKED_SELECTION_AST.encode()).hexdigest(),
                reference_projection_ast=reference_identity())
