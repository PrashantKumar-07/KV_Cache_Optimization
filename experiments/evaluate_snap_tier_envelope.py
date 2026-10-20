"""Frozen document-paired quality at shared owned-KV and width ceilings."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from audit_snap_tier_envelope import KV_CEILING, WIDTH_CEILING, SETTINGS
from audit_snap_tier_resources import distinct_storage_bytes, snapshot
from audit_unused_documents import document_key
from check_batched_generation import Trajectory, measure
from direct_shared_pool import DirectScorePolicy
import longbench_eval as bench
from provenance import atomic_json, read_jsonl, runtime_identity
from snapkv_adapter import SnapKVAdapter


class CpuEnvelopeStress(Trajectory, DirectScorePolicy):
    def __init__(self, record, **options):
        self.record = record
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)

    def _visible_tuple(self, past):
        result = super()._visible_tuple(past)
        tensors = [t for pair in result for t in pair]
        for cache in self.layer_caches.values():
            tensors.extend((cache.sram_k,cache.sram_v,cache.pool))
        owned = distinct_storage_bytes(tensors)
        width = max(pair[0].shape[-2] for pair in result)
        assert sum(owned.values()) <= KV_CEILING and width+1 <= WIDTH_CEILING
        self.record.update(final_owned_kv_storage_bytes=owned,
            max_owned_kv_storage_bytes=max(self.record.get('max_owned_kv_storage_bytes',0),sum(owned.values())),
            max_next_forward_rows=max(self.record.get('max_next_forward_rows',0),width+1),
            boundaries=self.frames)
        return result


@torch.inference_mode()
def snap_measure(model, tokenizer, item, task):
    prompt = bench.DATASET_TO_PROMPT[task].format(**item)
    if task not in bench.NO_CHAT_TASKS: prompt = f'[INST] {prompt} [/INST]'
    ids = tokenizer(prompt,return_tensors='pt').input_ids.to('cuda')
    if ids.shape[1]>31500: ids=torch.cat((ids[:,:15750],ids[:,-15750:]),1)
    count=ids.shape[1]; maximum=bench.DATASET_TO_MAXGEN[task]
    capacity=1025-maximum  # Before outputs: prompt + at most (cap-1) decode rows.
    cache=SnapKVAdapter(model,capacity=capacity,window=32,kernel=5)
    tokens=[]
    try:
        torch.cuda.reset_peak_memory_stats()
        with bench._attn_impl(model,'sdpa'):
            out=model(ids,past_key_values=cache,use_cache=True,
                      position_ids=torch.arange(count,device='cuda')[None])
        for step in range(maximum):
            token=out.logits[:,-1:].argmax(-1);value=token.item();tokens.append(value)
            if value==tokenizer.eos_token_id or step+1==maximum:break
            assert cache.layers[0].keys.shape[-2]+1<=WIDTH_CEILING
            out=model(token,past_key_values=cache,use_cache=True,
                      position_ids=torch.tensor([[count+step]],device='cuda'))
        boundary=snapshot(cache,None,len(tokens)-1)
        owned=boundary['distinct_owned_kv_storage_bytes']
        assert sum(owned.values())<=KV_CEILING
        return dict(prediction=tokenizer.decode(tokens,skip_special_tokens=True),
            generated_token_ids=tokens,max_gen=maximum,
            termination='eos' if tokens[-1]==tokenizer.eos_token_id else 'cap',
            generation=dict(prompt_tokens=count,generated_tokens=len(tokens)),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),snap_prompt_capacity=capacity,
            final_owned_kv_storage_bytes=owned,max_owned_kv_storage_bytes=sum(owned.values()),
            max_actual_forward_rows=max(min(count,capacity),min(count,capacity)+len(tokens)-1),
            instrumented_peak_gpu_bytes=torch.cuda.max_memory_allocated())
    finally:
        cache.close();del cache
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'Preserve evidence: {args.output}')
    manifest=json.loads(args.manifest.read_text())
    report=dict(passed=False,runtime=runtime_identity(),rows=[],
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))})
    try:
        assert manifest['arms']==['snap','cpu_direct'] and manifest['seed']==0
        requests=[]
        for task,meta in manifest['tasks'].items():
            path=Path(meta['slice_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['slice_sha256']
            rows=read_jsonl(path);assert len(rows)==20
            for item,sample,key in zip(rows,meta['selected_sample_ids'],meta['selected_normalized_context_sha256']):
                assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
                assert document_key(item)==key
                requests.append((task,item,sample,key))
        assert len(requests)==100 and len({r[3] for r in requests})==100
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(manifest['model'],torch_dtype=torch.bfloat16,
                device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(manifest['model'])
        report.update(gpu=torch.cuda.get_device_name(),expected_rows=200)
        for index,(task,item,sample,key) in enumerate(requests):
            for arm in (manifest['arms'] if index%2==0 else manifest['arms'][::-1]):
                torch.manual_seed(0)
                if arm=='snap':result=snap_measure(model,tokenizer,item,task)
                else:
                    resources={}
                    factory=lambda **options:CpuEnvelopeStress(resources,**options)
                    result=measure(model,tokenizer,item,task,0,'cpu_scores',31500,prompt_queries=True,
                        policy_cls=factory,policy_options=dict(sram_budget=991,stt_budget=1090,stt_expose_quota=32))
                    result.update(resources)
                prediction=result['prediction']
                if task in ('trec','triviaqa','samsum','lsht'):prediction=prediction.lstrip('\n').split('\n')[0]
                result['score']=100*max(bench.DATASET_TO_METRIC[task](prediction,answer,
                    all_classes=item.get('all_classes',[])) for answer in item['answers'])
                report['rows'].append(dict(request_index=index,task=task,sample_id=sample,
                    normalized_context_sha256=key,arm=arm,**result))
                atomic_json(args.output,report)
                print(index+1,'/100',task,arm,len(result['generated_token_ids']),result['score'],flush=True)
        report.update(passed=True,limitations='100 previously unused normalized documents/five tasks/one model/seed; standard NarrativeQA exhausted. Equal512MiBowned-KV/1024forward-row ceilings, actual usage and retention differ. Instrumented timings/peaks are descriptive. Source and paired bootstrap verification required; no physicalSTT, exactresource, throughput or broadcompletion claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        atomic_json(args.output,report)


if __name__=='__main__':main()
