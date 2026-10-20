"""Independent CPU page-layout reconstruction; no model/GPU/quality claim."""
from pathlib import Path
import ast
import hashlib
import json
import types
import torch

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'IceCache-bf16/IceCache/source/icecache'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output = BASE / 'independent_hnd_page_projection_proof.json'
    review = BASE / 'independent_projection_layout_review.json'
    assert not output.exists() and not review.exists(), 'Preserve prior evidence'
    torch.set_num_threads(1)
    torch.manual_seed(12345)
    source = SOURCE / 'adapter/modeling.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_icecache_prefill')
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'key' for t in n.targets)]
    assert len(assignments) == 1
    old_expr = assignments[0].value
    assert ast.unparse(old_expr) == 'key_states.reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]'
    old = compile(ast.Expression(old_expr), str(source), 'eval')
    fixed = compile(ast.parse('key_states.transpose(1, 2).reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]', mode='eval'), '<independent-layout-review>', 'eval')
    sources = [source, SOURCE / 'infer_state.py', SOURCE / 'kv_cache.py',
               BASE / 'M-DCI/src/dci_wrapper.cpp', BASE / 'M-DCI/src/dci.c',
               BASE / 'projection_layout_correction.py',
               BASE / 'check_dci_projection_contract.py']
    hashes = {str(p.relative_to(BASE)): sha(p) for p in sources}
    report = {'verified': False, 'CPU_only': True, 'torch_version': torch.__version__,
              'script_sha256': sha(Path(__file__)), 'sources_sha256': hashes,
              'cases': [], 'GPU_DCI_fullmodel_quality_verified': False}
    try:
        for length in (129, 256, 257, 1100):
            for heads in (1, 2, 8):
                page_size, dim = 16, 128
                keys = torch.randn(1, length, heads, dim, dtype=torch.bfloat16)
                pages = (length + page_size - 1) // page_size
                start, end = 2 * page_size, (pages - 2) * page_size
                padded = torch.zeros(1, pages * page_size, heads, dim, dtype=torch.bfloat16)
                padded[:, :length] = keys
                # KvPool layout_map=(0,2,1,3): [page,K/V,head,token,dim].
                hnd = padded[0].reshape(pages, page_size, heads, dim).permute(0, 2, 1, 3)
                all_kv = torch.stack((hnd, hnd + 1), dim=1)
                # prefill_backup_pages copies pages after sinks, before window.
                cpu_pages = all_kv[2:pages-2].clone()
                # Exact independent trace of the consumer's page permutation.
                indexed = cpu_pages.permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4).reshape(2, heads, -1, dim)[0]
                context = {'key_states': keys, 'state': types.SimpleNamespace(n_kv_heads=heads, head_dim=dim), 'start': start, 'end': end}
                corrected = eval(fixed, context)
                original = eval(old, context)
                reference = torch.stack([keys[0, start:end, h] for h in range(heads)])
                assert torch.equal(indexed, reference)
                assert torch.equal(corrected, indexed)
                assert torch.equal(original, indexed) == (heads == 1)
                report['cases'].append({'prompt_tokens': length, 'KV_heads': heads,
                    'indexed_tokens_per_head': end-start, 'corrected_exact': True,
                    'original_exact': heads == 1,
                    'original_mismatched_rows': int((original != indexed).any(-1).sum())})
        report['verified'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')
    findings = {'independent_review': True, 'page_layout_proof_verified': True,
        'sources_sha256': hashes, 'proof_path': output.name, 'proof_sha256': sha(output),
        'conclusion': 'Transpose correction preserves intended per-head indexed-key/projection correspondence for upstream B=1, n_groups=1, HND offload.',
        'compiled_index_test_review': 'Exact-axis debug-serial native-vs-precomputed token/leaf/offset comparison soundly isolates layout, including original multi-head index disagreement; not GPU retrieval or numerical-generalization proof.',
        'limitations': ['Batch sizes above1 are not supported by this expression; upstream offload already asserts B=1 and n_groups=1. Enforce those limits in the benchmark harness.', 'Expression/layout tests do not execute a complete corrected prefill.', 'Exact-axis projection deliberately excludes ordinary BF16-vs-FP32 projection rounding.'],
        'remaining_gates': ['Execute corrected function through actual GPU offload and compare projected rows with actual CPU indexed keys.', 'Validate DCI retrieval, returned page payloads and several decode steps, including non-page-aligned prompts.', 'Validate pretrained Mistral offload/model behavior before quality or performance comparisons.'],
        'baseline_accepted_for_comparison': False, 'SOTA_claim': False}
    review.write_text(json.dumps(findings, indent=2) + '\n')
    print(json.dumps({'verified_cases': len(report['cases']), 'proof': str(output), 'review': str(review)}))


if __name__ == '__main__':
    main()
