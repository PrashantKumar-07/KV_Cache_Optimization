"""Frozen six-task development regression and charged prefill screen."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import longbench_eval as bench
from bounded_pinned_pool import BoundedPoolPolicy
from check_batched_generation import measure
from check_bounded_pool_generation import Candidate
from evaluate_query_selection import PromptQueryCapture
from head_max_retention import use_head_max
from provenance import atomic_json, read_jsonl, runtime_identity


def prefill(model, ids):
    policy = BoundedPoolPolicy(sram_budget=991, stt_budget=1090, start_size=4,
        recent_size=128, kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75,
        stt_expose_quota=32)
    policy.prompt_query_score_mode = 'chunked'
    policy.prompt_query_chunk = 128
    capture = PromptQueryCapture(model, policy)
    out = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.inference_mode(), bench._attn_impl(model, 'sdpa'):
            out = model(ids, use_cache=True,
                        position_ids=torch.arange(ids.size(1), device=ids.device)[None])
            policy(bench.to_tuple_kv(out.past_key_values))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        positions = {str(i): dict(hot=c.sram_pos, cold=c.stt_pos)
                     for i,c in policy.layer_caches.items()}
        return dict(seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            placement_sha256=hashlib.sha256(json.dumps(positions, sort_keys=True).encode()).hexdigest(),
            reserved_host_tensor_bytes=sum(c.reserved_host_bytes for c in policy.layer_caches.values()),
            layer_counts={str(i): [len(c.sram_pos), len(c.stt_pos)] for i,c in policy.layer_caches.items()})
    finally:
        capture.close()
        policy.reset()
        del out, policy
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    base = args.directory
    output = base / 'result.json'
    if output.exists():
        raise FileExistsError(output)
    manifest_path, needles_path = base/'generation_manifest.json', base/'needles_manifest.json'
    manifest, needles = json.loads(manifest_path.read_text()), json.loads(needles_path.read_text())
    plan = json.loads((base/'plan.json').read_text())
    report = dict(passed=False, runtime=runtime_identity(), rows=[], prefill_rows=[],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        needles_sha256=hashlib.sha256(needles_path.read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((base/'plan.json').read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in list(Path('src').glob('*.py'))+list(Path('experiments').glob('*.py'))})
    try:
        prior_path = Path(plan['prior_control'])
        assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == plan['prior_control_sha256']
        prior = json.loads(prior_path.read_text())
        assert prior['passed'] and manifest['seeds'] == [0]
        controls = {r['sample_id']:r for r in prior['rows'] if r['arm']=='bounded' and r['seed']==0}
        requests = []
        for task, entry in manifest['tasks'].items():
            path = Path(entry['source_file'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['source_sha256']
            items = read_jsonl(path)
            for index, sample in zip(entry['selected_zero_based_indices'],entry['selected_sample_ids']):
                item = items[index]
                assert hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest() == sample
                requests.append((task,sample,item))
        assert len(requests)==12 and len(controls)==12
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.bfloat16,
                    device_map='cuda', attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'])
        report['gpu'] = torch.cuda.get_device_name()
        for index,(task,sample,item) in enumerate(requests):
            for arm in (('sum_heads','max_head') if index%2==0 else ('max_head','sum_heads')):
                with use_head_max() if arm=='max_head' else nullcontext():
                    result = measure(model,tokenizer,item,task,0,'cpu_scores',31500,
                        prompt_queries=True, policy_cls=Candidate,
                        policy_options=dict(sram_budget=991,stt_budget=1090,stt_expose_quota=32))
                assert result['reserved_host_tensor_bytes']==151257088
                if arm=='sum_heads':
                    for field in ('prediction','generated_token_ids','score','generation','max_gen','termination',
                        'prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters',
                        'prefill_query_shapes','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes',
                        'occupied_host_tensor_bytes'):
                        assert json.loads(json.dumps(result[field]))==controls[sample][field],(sample,field)
                report['rows'].append(dict(task=task,sample_id=sample,seed=0,arm=arm,**result))
                atomic_json(output,report)
                print(index+1,task,arm,'score',result['score'],flush=True)
        cases = {c['name']:c for c in needles['cases']}
        for name in plan['prefill_cases']:
            case = cases[name]
            assert hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest()==case['token_sha256']
            ids = torch.tensor(case['ids'],device='cuda',dtype=torch.long)[None]
            for arm in ('sum_heads','max_head'):
                with use_head_max() if arm=='max_head' else nullcontext():
                    prefill(model,ids)
            for block in range(3):
                for arm in ('sum_heads','max_head','max_head','sum_heads'):
                    with use_head_max() if arm=='max_head' else nullcontext():
                        result = prefill(model,ids)
                    report['prefill_rows'].append(dict(case=name,prompt_tokens=len(case['ids']),
                        token_sha256=case['token_sha256'],block=block,arm=arm,**result))
                    atomic_json(output,report)
                print('prefill',name,'block',block,'complete',flush=True)
            del ids
        assert len(report['rows'])==24 and len(report['prefill_rows'])==24
        report.update(passed=True,limitations='Reused development documents, one model/greedy seed. Scoring change affects initial cumulative importance and later eviction. Prefill includes actual-query capture/model/cache placement but not first decode, throughput or physical STT. No independent quality or SnapKV superiority claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(output,report)


if __name__=='__main__':
    main()
