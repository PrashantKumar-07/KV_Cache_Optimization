"""Preregistered three-arm evaluation of 100 unused normalized documents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from check_batched_generation import measure
import longbench_eval as bench
from provenance import atomic_json, read_jsonl, runtime_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    sources=['experiments/evaluate_fresh_documents.py','experiments/check_batched_generation.py',
        'experiments/evaluate_query_selection.py','experiments/batched_score_policy.py',
        'experiments/cpu_batched_cache.py','experiments/cpu_slot_cache.py','experiments/compare_cpu_batched.py']
    report=dict(passed=False,pid=os.getpid(),runtime=runtime_identity(),rows=[],
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    try:
        manifest=json.loads(args.manifest.read_text())
        verification=json.loads(args.manifest.with_name('verified_manifest.json').read_text())
        assert verification['frozen_source_slice_identity_exclusion_and_sampling_verified']
        assert verification['manifest_sha256']==report['manifest_sha256']
        assert manifest['seed']==0 and manifest['max_ctx']==31500
        requests=[]
        for task,meta in manifest['tasks'].items():
            path=Path(meta['slice_file'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['slice_sha256']
            items=read_jsonl(path)
            assert len(items)==20
            for item,sample,key in zip(items,meta['selected_sample_ids'],meta['selected_normalized_context_sha256']):
                assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
                requests.append((task,item,sample,key))
        assert len(requests)==100 and len({r[3] for r in requests})==100
        arms=['gpu_proxy','gpu_query','cpu_query']
        assert manifest['arms']==arms
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(manifest['model'],torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(manifest['model'])
        report.update(gpu=torch.cuda.get_device_name(),expected_rows=300)
        for index,(task,item,sample,key) in enumerate(requests):
            pair={}
            offset=index%3
            for arm in arms[offset:]+arms[:offset]:
                result=measure(model,tokenizer,item,task,0,'gpu' if arm.startswith('gpu') else 'cpu_scores',
                    manifest['max_ctx'],prompt_queries=arm!='gpu_proxy')
                prediction=result['prediction']
                if task in ('trec','triviaqa','samsum','lsht'):
                    prediction=prediction.lstrip('\n').split('\n')[0]
                result['score']=100*max(bench.DATASET_TO_METRIC[task](prediction,answer,
                    all_classes=item.get('all_classes',[])) for answer in item['answers'])
                shapes=result['prefill_query_shapes']
                if arm=='gpu_proxy':
                    assert not shapes
                else:
                    assert len(shapes)==32 and all(shape==[32,min(128,result['generation']['prompt_tokens']),128] for shape in shapes.values())
                pair[arm]=result
                report['rows'].append(dict(request_index=index,task=task,sample_id=sample,
                    normalized_context_sha256=key,arm=arm,**result))
                atomic_json(args.output,report)
                print(index+1,'/100',task,arm,len(result['generated_token_ids']),result['score'],flush=True)
            for field in ('prediction','generated_token_ids','score','generation','max_gen','termination',
                          'prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes'):
                assert pair['gpu_query'][field]==pair['cpu_query'][field],f'{task}/{sample}/{field}'
        assert len(report['rows'])==300
        report.update(passed=True,exact_cpu_gpu_actual_query_generation_state_and_trajectory=True,
            limitations='100 frozen normalized unused documents/five tasks/one model/seed; NarrativeQA corpus exhausted. Semantic duplicates/training overlap unverified. Instrumented times/peaks/physical resources descriptive only. No independent latency, matched-faithful-baseline or physical STT claim. Statistical quality decision requires separate preregistered paired bootstrap report.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output,report)


if __name__=='__main__':
    main()
