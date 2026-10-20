#!/usr/bin/env python3
"""CPU diagnostic comparing vendored SnapKV selection with the harness policy.

Synthetic inputs expose semantic differences; this is not a quality benchmark.
"""
import hashlib
import importlib.util
from pathlib import Path
import torch
from longbench_eval import SnapKVPolicy
from provenance import atomic_json


def main():
    root = Path(__file__).resolve().parents[1]
    upstream = root / 'sota/snapkv/snapkv/monkeypatch/snapkv_utils.py'
    spec = importlib.util.spec_from_file_location('vendored_snapkv_utils', upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    heads, groups, length, dim, budget, window = 2, 2, 64, 8, 16, 4
    k = torch.randn(1, heads, length, dim)
    q = torch.randn(1, heads * groups, length, dim)
    # Values encode original token positions so gather results are inspectable.
    v = torch.arange(length).float().view(1, 1, length, 1).expand_as(k)
    expanded_k, expanded_v = module.repeat_kv(k, groups), module.repeat_kv(v, groups)
    cluster = module.SnapKVCluster(window, budget, 5, 'avgpool')
    uk, uv = cluster.update_kv(expanded_k, q, expanded_v, None, groups)
    policy = SnapKVPolicy(budget=budget, start_size=0, recent_size=window)
    hk, hv = policy(((k, v),))[0]
    upstream_ids = uv[0, :, :, 0].long().tolist()
    harness_ids = hv[0, :, :, 0].long().tolist()
    assert uk.shape == (1, heads * groups, budget, dim)
    assert hk.shape == (1, heads, budget, dim)
    assert all(ids[-window:] == list(range(length-window, length)) for ids in upstream_ids)
    assert harness_ids[0] == harness_ids[1]
    assert len({tuple(ids) for ids in upstream_ids}) > 1
    changed_q = q.clone()
    changed_q[..., -window:, :] *= -1
    _, changed_v = cluster.update_kv(expanded_k, changed_q, expanded_v, None, groups)
    assert not torch.equal(uv, changed_v)
    report = {
        'scope': 'Synthetic CPU selection diagnostic; no model quality or speed claim',
        'upstream_file_sha256': hashlib.sha256(upstream.read_bytes()).hexdigest(),
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed': 0, 'upstream_token_ids_per_query_head': upstream_ids,
        'harness_token_ids_per_kv_head': harness_ids,
        'actual_query_changes_upstream_selection': True,
        'upstream_kv_payload_bytes': (uk.numel()+uv.numel())*uk.element_size(),
        'harness_kv_payload_bytes': (hk.numel()+hv.numel())*hk.element_size(),
        'differences': [
            'Upstream uses actual rotary Q; harness uses K as Q.',
            'Upstream masks future observation-window keys before softmax; harness does not.',
            'Upstream selects separately per expanded query head; harness averages heads.',
            'Upstream pools only pre-window scores; harness pools the whole sequence.',
            'Upstream decode appends KV; harness evicts with H2O to a fixed budget.',
            'Default harness pins sink tokens; upstream cluster has no sink pinning.'
        ]}
    output = root / 'results/baseline_fidelity/snapkv_probe.json'
    atomic_json(output, report)
    print(output)
    print('PASS: query sensitivity, per-head selection, recent window, expanded KV bytes')


if __name__ == '__main__':
    main()
