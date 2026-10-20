"""Faithful selector needle diagnostic with observational index callback only."""
import argparse
import ast
import copy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import MethodType

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from longbench_eval import _attn_impl
from provenance import atomic_json, runtime_identity
from snapkv_adapter import SnapKVAdapter


def observe_indices(cluster, callback):
    original = ast.parse(textwrap.dedent(inspect.getsource(cluster.update_kv)))
    count = 0

    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal count
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'indices' and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'expand'):
                count += 1
                return [node, ast.parse('self._audit_indices(indices[..., 0], q_len)').body[0]]
            return node

    altered = Insert().visit(copy.deepcopy(original))
    assert count == 1, 'Expected exactly one upstream index expansion'
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == '_audit_indices'):
                return None
            return node

    stripped = Strip().visit(copy.deepcopy(altered))
    assert ast.dump(stripped) == ast.dump(original), 'Observer changed selector computation'
    namespace = dict(cluster.update_kv.__func__.__globals__)
    exec(compile(ast.fix_missing_locations(altered), '<snapkv_index_observer>', 'exec'), namespace)
    cluster.update_kv = MethodType(namespace['update_kv'], cluster)
    cluster._audit_indices = callback
    return hashlib.sha256(ast.dump(altered).encode()).hexdigest()


@torch.inference_mode()
def measure(model, tokenizer, case):
    retained = []
    cache = SnapKVAdapter(model, capacity=1024, window=32, kernel=5)
    start, end = case['answer_span']

    def observe(indices, length):
        selected = indices.detach().cpu()[0]
        current = torch.arange(length - 32, length).expand(selected.shape[0], -1)
        positions = torch.cat((selected, current), dim=1)
        counts = ((positions >= start) & (positions < end)).sum(dim=1).tolist()
        retained.append(dict(layer=len(retained), query_heads=selected.shape[0],
            answer_span_tokens=end-start, retained_tokens_per_head=counts,
            complete_span_heads=sum(n == end-start for n in counts)))

    try:
        observer_hash = observe_indices(cache.cluster, observe)
        torch.cuda.reset_peak_memory_stats()
        ids = torch.tensor(case['ids'], device='cuda', dtype=torch.long).view(1, -1)
        with _attn_impl(model, 'sdpa'):
            out = model(ids, past_key_values=cache, use_cache=True,
                        position_ids=torch.arange(ids.shape[1], device='cuda').unsqueeze(0))
        generated = []
        for step in range(case['max_gen']):
            token = out.logits[:, -1:].argmax(-1)
            generated.append(token.item())
            if generated[-1] == tokenizer.eos_token_id or step + 1 == case['max_gen']:
                break
            out = model(token, past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[ids.shape[1] + step]], device='cuda'))
        torch.cuda.synchronize()
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        payload = sum(x.numel() * x.element_size() for layer in cache.layers for x in (layer.keys, layer.values))
        assert len(retained) == 32 and all(x['query_heads'] == 32 for x in retained)
        assert payload == (1024 + len(generated) - 1) * 32 * 32 * 128 * 2 * 2
        return dict(case=case['name'], prompt_tokens=len(case['ids']), token_sha256=case['token_sha256'],
            prediction=prediction, generated_token_ids=generated,
            answer_correct=case['answer'].casefold() in prediction.casefold(),
            retained_answer_by_layer=retained, complete_span_query_heads=sum(x['complete_span_heads'] for x in retained),
            final_owned_kv_payload_bytes=payload, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            observer_ast_sha256=observer_hash)
    finally:
        cache.close()
        del cache
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    report = dict(passed=False, scope=__doc__, runtime=runtime_identity(), rows=[],
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('experiments/check_snapkv_needles.py', 'experiments/snapkv_adapter.py', 'sota/snapkv/snapkv/monkeypatch/snapkv_utils.py')})
    try:
        manifest = json.loads(args.manifest.read_text())
        cases = [case for case in manifest['cases'] if case.get('oracle')]
        assert len(cases) == 6 and manifest['seed'] == 0
        for case in cases:
            assert hashlib.sha256(json.dumps(case['ids'], separators=(',', ':')).encode()).hexdigest() == case['token_sha256']
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16,
                                                   device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'])
        report['gpu'] = torch.cuda.get_device_name()
        for case in cases:
            row = measure(model, tokenizer, case)
            report['rows'].append(row)
            atomic_json(args.output, report)
            print(case['name'], row['answer_correct'], row['complete_span_query_heads'], 'complete-span heads', flush=True)
        report.update(passed=True, correct_cases=sum(x['answer_correct'] for x in report['rows']),
            resource_matched=False, limitations='Six reused synthetic probes, not independent quality data. Actual upstream selection with read-only index observer; callback distorts timing/peak allocation. Expanded query-head cache and growing decode differ from Tiered resources/width. No selector/default change or resource-matched superiority claim.')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print('diagnostic complete', flush=True)


if __name__ == '__main__':
    main()
