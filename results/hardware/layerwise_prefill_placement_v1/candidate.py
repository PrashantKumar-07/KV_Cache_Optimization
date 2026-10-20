"""Research-only layerwise placement; current prefill attention receives full KV."""
import ast
import copy
import hashlib
import inspect
import textwrap
import weakref
from transformers import DynamicCache
from longbench_eval import TieredKVPolicy
from bulk_promotion_policy import BulkPromotionPolicy


def placement_method():
    original = TieredKVPolicy._bifurcate
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    loops = [node for node in tree.body[0].body if isinstance(node, ast.For)]
    assert len(loops) == 1
    loop = loops[0]
    assert ast.unparse(loop.target) == '(layer_idx, (k, v))'
    assert ast.unparse(loop.iter) == 'enumerate(past_kv)'
    body = copy.deepcopy(loop.body)
    function = ast.parse('def place(self, layer_idx, k, v):\n    pass').body[0]
    function.body = body + [ast.Return(ast.Name(id='cache', ctx=ast.Load()))]
    assert ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=loop.body, type_ignores=[]), include_attributes=False)
    namespace = original.__globals__.copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 '<unchanged-ordinary-layer-placement>', 'exec'), namespace)
    return namespace['place'], hashlib.sha256(ast.dump(loop, include_attributes=False).encode()).hexdigest()


PLACE, PLACEMENT_AST_SHA256 = placement_method()


class EarlyPlacement:
    def __init__(self, **options):
        super().__init__(**options)
        self.placed_layers = []
        self.prefill_layers = 0
        self.prefill_sequence = None
        self.placement_rows = []

    def reset(self):
        super().reset()
        self.placed_layers = []
        self.prefill_sequence = None
        self.placement_rows = []

    def place(self, layer, key, value):
        if layer != len(self.placed_layers) or layer >= self.prefill_layers:
            raise ValueError('Require ordered, one-update-per-layer full prefill')
        if (key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1
                or key.shape[2] != self.prefill_sequence or layer not in self.prompt_q_by_layer):
            raise ValueError('Require full batch-one KV and actual prompt queries')
        original = getattr(self, 'cache_cls', None)
        if hasattr(self, 'direct_cache_cls'):
            def create(cfg):
                cache = self.direct_cache_cls(cfg)
                if self.manager is None:
                    self.manager = self.direct_pool_cls(cache, self.prefill_layers)
                self.manager.attach(cache)
                return cache
            self.cache_cls = create
        try:
            cache = PLACE(self, layer, key, value)
        finally:
            if original is None:
                self.__dict__.pop('cache_cls', None)
            else:
                self.cache_cls = original
        self.placed_layers.append(layer)
        self.placement_rows.append(dict(layer=layer, raw_shape=list(key.shape),
                                        hot_shape=list(cache.sram_k.shape)))
        return cache

    def _bifurcate(self, past):
        if self.placed_layers != list(range(self.prefill_layers)) or len(past) != self.prefill_layers:
            raise ValueError('Incomplete layerwise prefill placement')
        return self._visible_tuple(past)


class EarlyBulkPolicy(EarlyPlacement, BulkPromotionPolicy):
    pass


class EarlyGpuPolicy(EarlyPlacement, TieredKVPolicy):
    pass


class EarlyCache(DynamicCache):
    def __init__(self, policy):
        super().__init__()
        self.policy_ref = weakref.ref(policy)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        policy = self.policy_ref()
        if policy is None:
            raise RuntimeError('Layerwise policy owner expired')
        if policy.initialized:
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        cache = policy.place(layer_idx, key_states, value_states)
        super().update(cache.sram_k.unsqueeze(0), cache.sram_v.unsqueeze(0), layer_idx)
        # The current attention receives exactly the native full-prompt KV.
        return key_states, value_states


class PrefillLease:
    """Scoped serial batch-one hook; partial exceptions reset owned policy state."""
    def __init__(self, model, policy):
        self.policy_ref = weakref.ref(policy)
        self.layers = len(model.model.layers)
        self.handles = []
        try:
            self.handles.append(model.register_forward_pre_hook(self.before, with_kwargs=True))
            self.handles.append(model.register_forward_hook(self.after, with_kwargs=True, always_call=True))
        except BaseException:
            self.close()
            raise

    def before(self, model, positional, options):
        policy = self.policy_ref()
        if policy is None or policy.initialized:
            return
        ids = options.get('input_ids', positional[0] if positional else None)
        if (ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 1
                or options.get('past_key_values') is not None or not options.get('use_cache', True)
                or policy.placed_layers or options.get('inputs_embeds') is not None):
            raise ValueError('Require one full serial batch-one prefill without an existing cache')
        policy.prefill_layers, policy.prefill_sequence = self.layers, ids.shape[1]
        return positional, {**options, 'past_key_values': EarlyCache(policy)}

    def after(self, model, positional, options, output):
        policy = self.policy_ref()
        if output is None and policy is not None and not policy.initialized:
            policy.reset()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
