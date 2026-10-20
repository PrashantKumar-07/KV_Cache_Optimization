"""Experimental Mistral adapter for the vendored SnapKV prompt selector.

Expanded-head storage and growing decode cache intentionally follow upstream.
Use explicit absolute position_ids after compression. Single full prefill only.
"""
import importlib.util
from pathlib import Path
import torch
from transformers import DynamicCache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, rotate_half


def upstream_module():
    path = Path(__file__).resolve().parents[1] / 'sota/snapkv/snapkv/monkeypatch/snapkv_utils.py'
    spec = importlib.util.spec_from_file_location('snapkv_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SnapKVAdapter(DynamicCache):
    def __init__(self, model, capacity=1024, window=32, kernel=5, *, reuse_projection=False):
        super().__init__()
        if not 0 < window < capacity:
            raise ValueError('Require 0 < window < capacity')
        self.reference = upstream_module()
        self.cluster = self.reference.SnapKVCluster(window, capacity, kernel, 'avgpool')
        self.queries = {}
        self.original_groups = {}
        self.handles = []
        self.model = model
        self.reuse_projection = reuse_projection
        self.pending_positions = {}
        try:
            for i, layer in enumerate(model.model.layers):
                module = layer.self_attn
                self.original_groups[i] = module.num_key_value_groups
                module.num_key_value_groups = 1  # cache already expands KV heads
                self.handles.append(module.register_forward_pre_hook(self._capture(i), with_kwargs=True))
                if reuse_projection:
                    self.handles.append(module.q_proj.register_forward_hook(self._projected(i, module)))
                    self.handles.append(module.register_forward_hook(self._after(i), always_call=True))
        except BaseException:
            self.close()
            raise

    def _capture(self, index):
        def capture(module, args, kwargs):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            if self.get_seq_length(index) != 0:
                if hidden.shape[1] != 1:
                    raise ValueError('Only single-token decode after full prefill is supported')
                return
            if self.reuse_projection:
                if index in self.pending_positions:
                    raise RuntimeError('Nested SnapKV attention context is unsupported')
                self.pending_positions[index] = kwargs['position_embeddings']
                return
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q = module.q_proj(hidden).view(shape).transpose(1, 2)
            k = module.k_proj(hidden).view(shape).transpose(1, 2)
            cos, sin = kwargs['position_embeddings']
            self.queries[index] = apply_rotary_pos_emb(q, k, cos, sin)[0]
        return capture

    def _projected(self, index, attention):
        def projected(module, args, output):
            positions = self.pending_positions.pop(index, None)
            if positions is None:
                return
            q = output.view(*output.shape[:-1], -1, attention.head_dim).transpose(1, 2)
            cos, sin = positions
            self.queries[index] = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
        return projected

    def _after(self, index):
        def after(module, args, output):
            self.pending_positions.pop(index, None)
            self.queries.pop(index, None)
        return after

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        groups = self.original_groups[layer_idx]
        k = self.reference.repeat_kv(key_states, groups)
        v = self.reference.repeat_kv(value_states, groups)
        if self.get_seq_length(layer_idx) == 0:
            q = self.queries.pop(layer_idx)
            compressed = self.cluster.update_kv(k, q, v, None, groups)
            super().update(*compressed, layer_idx)
            # Prefill attention remains full; compression changes only future steps.
            return k, v
        return super().update(k, v, layer_idx)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for i, groups in self.original_groups.items():
            self.model.model.layers[i].self_attn.num_key_value_groups = groups
        self.original_groups.clear()
        self.handles.clear()
        self.queries.clear()
        self.pending_positions.clear()
