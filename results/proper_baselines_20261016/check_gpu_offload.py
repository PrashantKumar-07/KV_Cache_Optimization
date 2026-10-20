"""Prospective tiny BF16 offload correctness check; never a benchmark.

Parent process starts one fresh child per context. Root must review the pinned
plan and reserve GPU1 before execution. No upstream source is modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback

BASE = Path(__file__).resolve().parent
PLAN = BASE / 'offload_preflight_plan.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def check_plan():
    plan = json.loads(PLAN.read_text())
    for name, expected in plan['sources_sha256'].items():
        assert sha(BASE / name) == expected, name
    assert plan['contexts'] == [513, 529]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'Reserve physical GPU1 explicitly'
    return plan


def worker(length):
    import copy
    import types
    import torch
    from icecache import adapter
    from icecache.infer_state import InferState
    from transformers import MistralConfig, MistralForCausalLM
    from projection_layout_correction import corrected_projection

    plan = check_plan()
    output = BASE / f'offload_preflight_context_{length}.json'
    assert length in plan['contexts'] and not output.exists()
    report = {'verified': False, 'execution_complete': False, 'prompt_tokens': length,
              'plan_sha256': sha(PLAN), 'index_builds': [], 'queries': [], 'decode': [],
              'scope': plan['scope'], 'torch_version': torch.__version__}
    try:
        torch.manual_seed(plan['seed'])
        torch.cuda.manual_seed_all(plan['seed'])
        torch.set_num_threads(1)
        device = torch.device('cuda')
        config = MistralConfig(**plan['model_config'])
        config._attn_implementation = 'sdpa'
        reference = MistralForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
        patched = copy.deepcopy(reference)
        weights = hashlib.sha256()
        for name, tensor in reference.state_dict().items():
            weights.update(name.encode()); weights.update(bytes.fromhex(tensor_hash(tensor)))
            assert torch.equal(tensor, patched.state_dict()[name])
        report['model_weights_sha256'] = weights.hexdigest()
        report['model_config'] = config.to_dict()
        state = InferState(2, 32, 8, 128, 16, torch.bfloat16, device,
            page_budgets=16, page_topks=8, n_max_pages=256, n_max_cpu_pages=2048,
            n_sink_pages=2, n_win_pages=2, debug=True,
            n_reuse_layers=0, n_prefetch_layers=0)
        state.parallel_level = 0
        original_build = state._DCI_first_call
        original_query = state._DCI_query
        captured = {}

        def check_payload(layer, selected=None):
            db = state.dci_db[layer]
            leaves, offsets = db.get_token2node()
            keys, values = captured[layer]
            assert leaves.shape == offsets.shape == keys.shape[:2]
            counts = db.get_num_leaves()
            points = db.get_num_points()
            assert torch.equal(points, torch.full_like(points, keys.shape[1]))
            cache = state.cpu_kv_caches[layer]
            checked = 0
            for head in range(8):
                assert ((leaves[head] >= 0) & (leaves[head] < counts[head])).all()
                assert ((offsets[head] >= 0) & (offsets[head] < 16)).all()
                mask = torch.ones(keys.shape[1], dtype=torch.bool)
                if selected is not None:
                    mask = torch.isin(leaves[head], selected[head].cpu())
                indices = mask.nonzero().flatten()
                assert indices.numel() > 0
                for token in indices.tolist():
                    leaf, offset = int(leaves[head, token]), int(offsets[head, token])
                    page = cache[0, leaf]
                    assert torch.equal(page[0, head, offset], keys[head, token])
                    assert torch.equal(page[1, head, offset], values[head, token])
                checked += indices.numel()
            return {'exact_KV_token_rows': checked, 'token_leaf_map_sha256': tensor_hash(leaves),
                    'token_offset_map_sha256': tensor_hash(offsets), 'leaf_counts': counts.tolist()}

        @torch.inference_mode()
        def build(self, batch, layer, query, keys, values, projected):
            assert batch == 0 and self.n_groups == 1 and layer not in captured
            assert keys.shape == values.shape == (8, ((length+15)//16-4)*16, 128)
            assert projected.shape == (8, keys.shape[1], 1)
            assert torch.isfinite(keys).all() and torch.isfinite(values).all()
            # C data_projection mutates projected in place, so verify beforehand.
            untouched = projected.clone()
            expected = (keys.to(device) @ self.proj_vec[:-1]).float().reshape(8, -1, 1).cpu()
            assert torch.equal(untouched, expected), 'GPU projection/indexed-key mismatch'
            captured[layer] = (keys.clone(), values.clone())
            item = {'layer': layer, 'indexed_tokens_per_head': keys.shape[1],
                    'projection_exact_before_C_mutation': True,
                    'projection_sha256': tensor_hash(untouched),
                    'indexed_keys_sha256': tensor_hash(keys), 'indexed_values_sha256': tensor_hash(values)}
            report['index_builds'].append(item)
            result = original_build(batch, layer, query, keys, values, projected)
            item.update(check_payload(layer))
            item['index_build_and_CPU_rearrangement_exact'] = True
            return result

        @torch.inference_mode()
        def query(self, batch, layer, query_states):
            item = {'layer': layer, 'query_shape': list(query_states.shape), 'complete': False}
            report['queries'].append(item)
            result = original_query(batch, layer, query_states)
            assert result is not None
            evicted, recalled, number = result
            selected = self.selected_page_idx[layer]
            counts = self.dci_db[layer].get_num_leaves()
            assert selected.shape == (8, 4)
            assert ((selected >= 0) & (selected < counts[:, None])).all()
            assert recalled.shape == evicted.shape
            assert recalled.shape[0] == number.numel() == 8
            assert torch.equal((recalled >= 0).sum(1).cpu(), number.cpu())
            for head in range(8):
                valid = recalled[head] >= 0
                assert (recalled[head, valid].cpu() < counts[head]).all()
                assert ((evicted[head, valid] >= 0) & (evicted[head, valid] < 256)).all()
            item.update({'selected_leaf_indices': selected.cpu().tolist(),
                         'recalled_leaf_indices': recalled.cpu().tolist(),
                         'evicted_GPU_pool_pages': evicted.cpu().tolist(),
                         'recalled_count_per_head': number.cpu().tolist(), **check_payload(layer, selected),
                         'complete': True})
            return result

        state._DCI_first_call = types.MethodType(build, state)
        state._DCI_query = types.MethodType(query, state)
        adapter.enable_icecache(patched, dtype=torch.bfloat16, device=device, infer_state=state)
        ids = torch.randint(0, 128, (1, length), device=device)
        teachers = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
        report['input_ids_sha256'] = tensor_hash(ids)
        report['teacher_ids_sha256'] = tensor_hash(teachers)
        with torch.inference_mode(), corrected_projection() as identity:
            report['correction'] = identity
            positions = torch.arange(length, device=device)[None]
            expected = reference(ids, use_cache=False, position_ids=positions, logits_to_keep=1).logits
            actual = patched(ids, use_cache=True, position_ids=positions, logits_to_keep=1).logits
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
            relative = ((actual.float()-expected.float()).norm()/expected.float().norm().clamp(min=1e-12)).item()
            torch.testing.assert_close(actual.float(), expected.float(), atol=.08, rtol=.02)
            assert relative <= .01
            report['prefill'] = {'relative_L2': relative,
                'max_abs_error': (actual.float()-expected.float()).abs().max().item(),
                'argmax_equal': bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))}
            assert len(captured) == len(report['index_builds']) == 2
            assert all(x.get('index_build_and_CPU_rearrangement_exact') for x in report['index_builds'])
            for step in range(6):
                position = torch.tensor([[length+step]], device=device)
                logits = patched(teachers[:, step:step+1], use_cache=True, position_ids=position, logits_to_keep=1).logits
                assert torch.isfinite(logits).all()
                torch.cuda.synchronize()
                report['decode'].append({'teacher_id': int(teachers[0, step]),
                    'absolute_position': length+step, 'all_finite': True,
                    'argmax_token': int(logits.argmax(-1).item()), 'logits_sha256': tensor_hash(logits)})
            assert len(report['queries']) == 12, 'Actual DCI query required for every layer/decode'
            assert all(x['complete'] for x in report['queries'])
        report.update(verified=True, execution_complete=True,
            GPU_pool_bytes=state._pool.buffer.numel()*state._pool.buffer.element_size(),
            CPU_pool_bytes=state._cpu_pool.buffer.numel()*state._cpu_pool.buffer.element_size(),
            full_model_or_recall_quality_guarantee=False)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--context', type=int)
    args = parser.parse_args()
    if args.context is not None:
        worker(args.context); return
    plan = check_plan()
    output = BASE / 'offload_preflight.json'
    assert not output.exists()
    assert all(not (BASE/f'offload_preflight_context_{n}.json').exists() for n in plan['contexts'])
    report = {'verified': False, 'execution_complete': False, 'plan_sha256': sha(PLAN),
              'children': [], 'scope': plan['scope']}
    try:
        for length in plan['contexts']:
            log = BASE/f'offload_preflight_context_{length}.log'
            with log.open('x') as stream:
                child = subprocess.run([sys.executable, str(Path(__file__)), '--context', str(length)],
                    cwd=BASE, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            item = {'context': length, 'exit_code': child.returncode, 'log_sha256': sha(log)}
            report['children'].append(item)
            assert child.returncode == 0, item
            path = BASE/f'offload_preflight_context_{length}.json'
            result = json.loads(path.read_text()); item['result_sha256'] = sha(path)
            assert result['verified'] and result['execution_complete']
            assert result['plan_sha256'] == sha(PLAN)
        report.update(verified=True, execution_complete=True, no_SOTA_or_full7B_claim=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
