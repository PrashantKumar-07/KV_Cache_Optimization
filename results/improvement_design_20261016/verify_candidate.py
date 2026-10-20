"""CPU-only independent reference rescoring and fixed development gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from collections import defaultdict
import math

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('independent_full_verifier',ROOT/'results/full_evaluation_20261015/verify_results.py')
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def sha(path):
    return reference.sha(path)


def main():
    plan=json.loads((BASE/'plan.json').read_text())
    plan_hash=sha(BASE/'plan.json')
    expected={BASE/'records'/f"{r['task']}_{r['source_zero_based_index']:04d}.json" for r in plan['rows']}
    present=set((BASE/'records').glob('*_*.json'))
    assert present==expected, f'Incomplete or extra records: {len(present)}/{len(expected)}'
    completion=json.loads((BASE/'records/complete.json').read_text())
    assert completion['execution_complete'] and completion['plan_sha256']==plan_hash and completion['records']==135
    assert not list((BASE/'errors').glob('*.json')), 'Recorded errors require independent review before verification'
    for path,digest in plan['source_sha256'].items():
        assert sha(ROOT/path)==digest, f'Source identity mismatch: {path}'
    from provenance import read_jsonl, model_artifact_identity
    from transformers import AutoTokenizer
    assert model_artifact_identity(plan['model'])==plan['model_identity']
    runtime=json.loads((BASE/'records/runtime.json').read_text())
    assert runtime['plan_sha256']==plan_hash and runtime['runtime']==plan['runtime']
    tokenizer=AutoTokenizer.from_pretrained(plan['model'])
    sources={}
    for task,path in plan['dataset_files'].items():
        assert sha(ROOT/path)==plan['dataset_sha256'][task]
        sources[task]=read_jsonl(ROOT/path)
    metrics,metric_source=reference.reference_metrics()
    config=json.loads((Path(plan['model'])/'config.json').read_text())
    rows=[]; raw_hashes={}
    for row in plan['rows']:
        task,index=row['task'],row['source_zero_based_index']
        item=sources[task][index]
        assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==row['sample_id']
        records={}
        for arm,entry in row['controls'].items():
            path=ROOT/entry['path']
            assert sha(path)==entry['sha256']
            records[arm]=json.loads(path.read_text())
            raw_hashes[str(path.relative_to(ROOT))]=sha(path)
        path=BASE/'records'/f'{task}_{index:04d}.json'
        records['candidate']=json.loads(path.read_text())
        raw_hashes[str(path.relative_to(ROOT))]=sha(path)
        candidate=records['candidate']; old=records['tiered_hot992']
        prompt=plan['protocol']['prompt_templates'][task].format(**item)
        if task not in plan['protocol']['no_chat_tasks']:
            prompt='[INST] '+prompt+' [/INST]'
        ids=tokenizer(prompt,return_tensors='pt').input_ids
        if ids.shape[1]>31500:
            import torch
            ids=torch.cat((ids[:,:15750],ids[:,-15750:]),dim=1)
        assert hashlib.sha256(ids.numpy().tobytes()).hexdigest()==candidate['input_ids_sha256']
        assert hashlib.sha256(prompt.encode()).hexdigest()==candidate['prompt_sha256']
        assert candidate['plan_sha256']==plan_hash
        assert candidate['source_index']==index and candidate['controls']==row['controls']
        assert candidate['document_group_sha256']==row['document_group_sha256']
        assert candidate['unchanged_input_and_initial_geometry_and_first_token']
        for field in ('input_ids_sha256','prompt_sha256','prompt_tokens','untruncated_prompt_tokens',
                      'geometry_after_prefill_before_first_decode'):
            assert candidate[field]==old[field], field
        assert candidate['generated_token_ids'][0]==old['generated_token_ids'][0]
        scores={}
        for arm,record in records.items():
            reference.assert_finite(record)
            assert record['execution_complete'] and record['sample_id']==row['sample_id']
            assert record['task']==task and record['answers']==item['answers']
            assert record['all_model_logits_finite'] and record['hooks_and_groups_restored']
            assert len(record['generated_token_ids'])==record['generated_tokens']
            assert 1<=record['generated_tokens']<=plan['protocol']['max_gen'][task]
            metric=plan['protocol']['metrics'][task]
            prediction=record['prediction']
            assert tokenizer.decode(record['generated_token_ids'],skip_special_tokens=True)==prediction
            scored=prediction.lstrip('\n').split('\n')[0] if task in ('triviaqa','trec','samsum','lsht') else prediction
            assert scored==record['scored_prediction']
            score=100*max(metrics[metric](scored,answer,all_classes=item.get('all_classes',[])) for answer in item['answers'])
            assert math.isclose(score,record['score'],abs_tol=1e-9)
            reference.verify_geometry(record, 'tiered_hot992' if arm=='candidate' else arm,
                record['prompt_tokens'], config['num_hidden_layers'],config['num_key_value_heads'],
                config['num_attention_heads'],config['hidden_size']//config['num_attention_heads'])
            scores[arm]=score
        rows.append(dict(task=task,sample_id=row['sample_id'],document_group_sha256=row['document_group_sha256'],scores=scores))
    vs_old=reference.primary_bootstrap(rows,'candidate','tiered_hot992',draws=10000,seed=0)
    vs_snap=reference.primary_bootstrap(rows,'candidate','snapkv_faithful',draws=10000,seed=0)
    gate=dict(gain_vs_old_at_least_1=vs_old['point_estimate']>=1,
        lower95_vs_old_above_0=vs_old['ci95'][0]>0,
        no_task_mean_loss_beyond_5=all(v['paired_mean_difference']>=-5 for v in vs_old['per_task_descriptive'].values()),
        point_gain_vs_faithful_snap_above_0=vs_snap['point_estimate']>0)
    result=dict(verification_passed=True,plan_sha256=plan_hash,rows=135,predictions_rescored=405,
        raw_files_sha256=raw_hashes,metric_reference_sha256=sha(metric_source),
        verifier_sha256=sha(__file__),vs_old=vs_old,vs_faithful_snap=vs_snap,
        registered_gate=gate,advance=all(gate.values()),
        decision='Advance only to separate resource and independent-data gates' if all(gate.values()) else 'REJECT and STOP candidate expansion',
        scope='Consumed original135 development only. Faithful Snap comparison has disclosed unequal cache geometry. No SOTA or independent quality claim.')
    target=BASE/'verified_candidate.json'
    assert not target.exists(), 'Preserve existing verification'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_files_sha256'},indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['--self-test']:
        reference.self_test()
        print('Reference metric, paired document bootstrap and finite geometry fixtures PASS')
    else:
        assert not sys.argv[1:]
        main()
