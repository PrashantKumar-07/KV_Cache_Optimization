"""Durable single-candidate development screen; original135 IDs only."""
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src'), str(BASE)]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from exposure_prior import ExposureCalibratedPriorPolicy
from provenance import atomic_json, model_artifact_identity, runtime_identity, read_jsonl

spec = importlib.util.spec_from_file_location('frozen_full_runner', ROOT/'results/full_evaluation_20261015/run_full.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


def sha(path):
    return full.sha(path)


def candidate_factory(arm):
    assert arm == 'tiered_hot992'
    return ExposureCalibratedPriorPolicy(sram_budget=992, stt_budget=2048,
        stt_expose_quota=32, start_size=4, recent_size=128,
        kv_dtype=torch.bfloat16, sttram_bifurcation_frac=.75)


def main():
    plan_path=BASE/'plan.json'
    plan=json.loads(plan_path.read_text())
    plan_hash=sha(plan_path)
    assert len(plan['rows']) == plan['candidate_count'] == 135
    for path,digest in plan['source_sha256'].items():
        assert sha(ROOT/path) == digest, f'Source mismatch: {path}'
    assert runtime_identity() == plan['runtime']
    assert model_artifact_identity(plan['model']) == plan['model_identity']
    populations={}
    for task,path in plan['dataset_files'].items():
        assert sha(ROOT/path) == plan['dataset_sha256'][task]
        populations[task]=read_jsonl(ROOT/path)
    for row in plan['rows']:
        for control in row['controls'].values():
            assert sha(ROOT/control['path']) == control['sha256']
    destination=BASE/'records'
    destination.mkdir(exist_ok=True)
    with (destination/'.running.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        torch.manual_seed(0)
        with torch.inference_mode():
            model=AutoModelForCausalLM.from_pretrained(plan['model'], torch_dtype=torch.bfloat16,
                device_map='cuda', attn_implementation='eager').eval()
            probe=model(torch.tensor([[1]],device='cuda'),use_cache=False,output_attentions=True,
                **full.bench.prefill_logit_kwargs(model, True))
            assert torch.isfinite(probe.logits).all()
            del probe
            tokenizer=AutoTokenizer.from_pretrained(plan['model'])
            atomic_json(destination/'runtime.json',dict(plan_sha256=plan_hash,
                runtime=runtime_identity(), GPU=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
            for row in plan['rows']:
                task,index=row['task'],row['source_zero_based_index']
                item=populations[task][index]
                sid=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
                assert sid == row['sample_id']
                reference=json.loads((ROOT/row['controls']['tiered_hot992']['path']).read_text())
                prompt=full.bench.DATASET_TO_PROMPT[task].format(**item)
                if task not in full.bench.NO_CHAT_TASKS:
                    prompt='[INST] '+prompt+' [/INST]'
                ids=tokenizer(prompt,return_tensors='pt').input_ids
                untruncated=ids.shape[1]
                if untruncated>31500:
                    ids=torch.cat((ids[:,:15750],ids[:,-15750:]),dim=1)
                tokenhash=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                assert tokenhash == reference['input_ids_sha256']
                assert hashlib.sha256(prompt.encode()).hexdigest() == reference['prompt_sha256']
                path=destination/f'{task}_{index:04d}.json'
                identity=dict(plan_sha256=plan_hash,task=task,source_index=index,sample_id=sid,
                    input_ids_sha256=tokenhash, prompt_sha256=reference['prompt_sha256'],
                    untruncated_prompt_tokens=untruncated,document_group_sha256=row['document_group_sha256'],
                    arm='exposure_calibrated_prior',controls=row['controls'])
                if path.exists():
                    previous=json.loads(path.read_text())
                    assert previous['execution_complete'] and all(previous[k]==v for k,v in identity.items())
                    continue
                previous_factory=full.policy_for
                try:
                    full.policy_for=candidate_factory
                    result=full.run_request(model,tokenizer,ids.to('cuda'),task,item,'tiered_hot992')
                    assert result['generated_token_ids'][0] == reference['generated_token_ids'][0]
                    assert result['geometry_after_prefill_before_first_decode'] == reference['geometry_after_prefill_before_first_decode']
                    assert not path.exists()
                    atomic_json(path,dict(identity,**result,execution_complete=True,
                        unchanged_input_and_initial_geometry_and_first_token=True))
                    print(task,index,result['score'],flush=True)
                except BaseException:
                    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                    atomic_json(BASE/'errors'/f'{task}_{index}_{stamp}.json',
                        dict(identity,execution_complete=False,traceback=traceback.format_exc()))
                    raise
                finally:
                    full.policy_for=previous_factory
                    gc.collect(); torch.cuda.empty_cache()
                del ids
            atomic_json(destination/'complete.json',dict(plan_sha256=plan_hash,
                execution_complete=True,records=135))


if __name__ == '__main__':
    main()
