"""Launch-only L40S prefill portability screen against causal GQA attention."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from flexicache_launch import StageOnePrefill
from provenance import atomic_json, runtime_identity


COMMIT = 'ab0f495b10fc6509c1637af35779b29121fb7d3f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT or subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Expected unchanged frozen release')
    source = args.upstream / 'vllm/attention/ops/prefix_prefill.py'
    report = {'scope': __doc__, 'passed': False, 'upstream_commit': revision,
              'kernel_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'adapter_sha256': hashlib.sha256(Path(__file__).with_name('flexicache_launch.py').read_bytes()).hexdigest(),
              'runtime': runtime_identity(), 'checks': []}
    try:
        from vllm.attention.ops import prefix_prefill as module
        if Path(module.__file__).resolve() != source.resolve():
            raise ValueError('Wrong kernel module import')
        adapter = StageOnePrefill(module.fwd_kernel_flexicache)
        module.fwd_kernel_flexicache = adapter
        report['gpu'] = torch.cuda.get_device_name()
        torch.manual_seed(0)
        for prefix, count in ((0, 512), (0, 1440), (64, 512)):
            heads, kv_heads, dim, page, x = 32, 8, 128, 16, 8
            total = prefix + count
            query = torch.randn(count, heads, dim, device='cuda', dtype=torch.bfloat16)
            keys, values = (torch.randn(total, kv_heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(2))
            blocks = max(1, prefix // page)
            table = torch.arange(kv_heads * blocks, device='cuda', dtype=torch.int32).reshape(1, kv_heads, blocks)
            k_cache = torch.zeros((kv_heads * blocks, dim // x, page, x), device='cuda', dtype=torch.bfloat16)
            v_cache = torch.zeros((kv_heads * blocks, dim, page), device='cuda', dtype=torch.bfloat16)
            for head in range(kv_heads):
                for block in range(prefix // page):
                    k_cache[head * blocks + block] = keys[block * page:(block + 1) * page, head].reshape(page, dim // x, x).permute(1, 0, 2)
                    v_cache[head * blocks + block] = values[block * page:(block + 1) * page, head].T
            output = torch.empty_like(query)
            one = torch.ones((), device='cuda')
            module.context_attention_fwd(query, keys[prefix:].contiguous(), values[prefix:].contiguous(), output,
                'auto', k_cache, v_cache, table,
                torch.tensor([0, count], device='cuda', dtype=torch.int32),
                torch.tensor([total], device='cuda', dtype=torch.int32), count, one, one,
                enable_flexicache=True)
            torch.cuda.synchronize()
            q = query.transpose(0, 1).float()
            k = keys.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).float()
            v = values.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).float()
            scores = torch.matmul(q, k.transpose(1, 2)) / dim**.5
            causal = torch.arange(total, device='cuda')[None, :] <= prefix + torch.arange(count, device='cuda')[:, None]
            reference = torch.matmul(scores.masked_fill(~causal, float('-inf')).softmax(-1), v).transpose(0, 1)
            error = (output.float() - reference).abs()
            max_error, mean_error = error.max().item(), error.mean().item()
            assert torch.isfinite(output).all() and max_error <= .03125 and mean_error <= .001, (prefix, count, max_error, mean_error)
            report['checks'].append({'prefix_tokens': prefix, 'new_tokens': count, 'max_abs_error': max_error,
                                     'mean_abs_error': mean_error, 'launch': adapter.calls[-1]})
            print(report['checks'][-1], flush=True)
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps({'passed': report['passed']}))


if __name__ == '__main__':
    main()
