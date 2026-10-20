"""Exact bulk CPU promotion metadata access with original unsupported fallback."""
import ast
import hashlib
import inspect
import textwrap
import types
import numpy as np
import torch
from longbench_eval import TieredKVPolicy
from tiered_kv_cache import TieredKVCache


def bulk_shadow_method(method, owner):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    before = ast.dump(tree, include_attributes=False)
    changed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == 'stt_shadow':
            assert isinstance(node.value, ast.Name) and node.value.id == owner
            changed.append(node)
    class Replace(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if node in changed: return ast.Name(id='shadow', ctx=ast.Load())
            return self.generic_visit(node)
    tree = Replace().visit(tree)
    setup = ast.parse(f'shadow = {owner}.stt_shadow.numpy()').body[0]
    if method.__name__ == '_promotion_plan':
        branch = next(n for n in tree.body[0].body if isinstance(n, ast.If))
        assert ast.unparse(branch.test) == 'stt_attn'
        branch.body.insert(0, setup)
    else:
        tree.body[0].body.insert(0, setup)
    after = ast.dump(tree, include_attributes=False)
    namespace = method.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree), '<bulk-CPU-shadow-access>', 'exec'), namespace)
    return namespace[method.__name__], dict(control=hashlib.sha256(before.encode()).hexdigest(),
        candidate=hashlib.sha256(after.encode()).hexdigest(), replacements=len(changed))


# Keep control flow, sorting, floor and fallback semantics in their original AST.
BULK_PLAN, PLAN_AST = bulk_shadow_method(TieredKVPolicy._promotion_plan, 'cache')
BULK_ROWS, ROWS_AST = bulk_shadow_method(TieredKVCache.promotion_rows, 'self')


def eligible(cache):
    m = len(cache.stt_pos)
    return all(t.device.type == 'cpu' and t.shape == (m,) and t.dtype == dtype
               for t, dtype in ((cache.stt_shadow, torch.bool),
                   (cache.stt_was_vram_resident, torch.bool), (cache.stt_peek_streak, torch.float32)))


def resolve(cache, indices):
    if not eligible(cache): return TieredKVCache.resolve_promotion_indices(cache, indices)
    m = len(cache.stt_pos)
    if not m: return [], []
    shadow = cache.stt_shadow.numpy()
    indices = list(dict.fromkeys(i for i in indices if 0 <= i < m and not shadow[i]))
    if not indices:
        cache.stt_peek_streak.zero_()
        return [], []
    mask = np.zeros(m, dtype=np.bool_)
    mask[indices] = True
    streak = cache.stt_peek_streak.numpy()
    cache.stt_peek_streak = torch.from_numpy(np.where(mask, streak + np.float32(1), np.zeros_like(streak)))
    idx = np.asarray(indices, dtype=np.int64)
    allowed = ((~cache.stt_was_vram_resident.numpy()[idx]) |
        (cache.stt_peek_streak.numpy()[idx] >= cache._HYSTERESIS_MIN_STREAK)).tolist()
    return ([i for i, ok in zip(indices, allowed) if ok],
            [i for i, ok in zip(indices, allowed) if not ok])


def plan(policy, layer, cache):
    method = BULK_PLAN if eligible(cache) else TieredKVPolicy._promotion_plan
    return method(policy, layer, cache)


def rows(cache, hits):
    method = BULK_ROWS if eligible(cache) else TieredKVCache.promotion_rows
    return method(cache, hits)


def fixture(frame, fast):
    policy = types.SimpleNamespace(promotion_budget=frame['budget'],
        _stt_attn=frame['attention'], _stt_attn_idx=frame['attention_indices'])
    caches = {}
    for layer, values in frame['layers'].items():
        cache = types.SimpleNamespace(stt_pos=list(values['positions']),
            cfg=types.SimpleNamespace(inclusive=values['inclusive'], stt_live_floor=values['floor']),
            _HYSTERESIS_MIN_STREAK=values['hysteresis'],
            **{n: values[n].clone() for n in ('stt_shadow','stt_was_vram_resident','stt_peek_streak')})
        cache.resolve_promotion_indices = types.MethodType(resolve if fast else TieredKVCache.resolve_promotion_indices, cache)
        caches[layer] = cache
    return policy, caches


def run(policy, caches, fast):
    planner = plan if fast else TieredKVPolicy._promotion_plan
    row_planner = rows if fast else TieredKVCache.promotion_rows
    plans = {l: planner(policy, l, c) for l, c in caches.items()}
    selections, promotion_rows = {}, {}
    for layer, (hits, _) in plans.items():
        if hits:
            indices, _ = row_planner(caches[layer], hits)
            if indices: selections[layer] = indices
            # The production promote path repeats the row/floor check after prefetch.
            promotion_rows[layer] = row_planner(caches[layer], hits)
    return plans, selections, promotion_rows
