"""Verify usual-budget generation, source/data identity and request isolation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
import longbench_eval as bench
from provenance import read_jsonl
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((BASE/'generation_manifest.json').read_text())
    plan = json.loads((BASE/'plan.json').read_text())
    generation = json.loads((BASE/'generation.json').read_text())
    interleaved = json.loads((BASE/'interleaved.json').read_text())
    frozen = json.loads((BASE/'source_manifest.json').read_text())['source_sha256']
    assert manifest['hot']==plan['hot']==1024 and manifest['cold']==plan['cold']==2048
    assert manifest['seeds']==[0] and generation['passed'] and interleaved['passed']
    assert generation['manifest_sha256']==sha(BASE/'generation_manifest.json')
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for raw in (generation,interleaved):
            for name,digest in raw['source_sha256'].items():
                assert digest==frozen[name]
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
        driver = archive.extractfile('experiments/check_bounded_pool_generation.py').read().decode()
    with tarfile.open(ROOT/'results/baseline_fidelity/head_max_broader_dev_20261002/source_v2.tar.gz') as archive:
        old_driver = archive.extractfile('experiments/check_bounded_pool_generation.py').read().decode()
    previous,new = ast.parse(old_driver),ast.parse(driver)
    for name in ('Control','Candidate'):
        assert ast.dump(next(n for n in previous.body if isinstance(n,ast.ClassDef) and n.name==name)) == ast.dump(
            next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name==name))
    assert (1090+991+1)*131072==272891904
    assert (1090+32+32)*131072==151257088
    tokenizer = AutoTokenizer.from_pretrained(str((ROOT/manifest['model']).resolve()))
    pairs = []
    for task,entry in manifest['tasks'].items():
        source = ROOT/entry['source_file']
        assert sha(source)==entry['source_sha256']
        data = read_jsonl(source)
        for index,sample in zip(entry['selected_zero_based_indices'],entry['selected_sample_ids']):
            item = data[index]
            assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
            rows=[r for r in generation['rows'] if r['sample_id']==sample]
            assert len(rows)==2 and {r['arm'] for r in rows}=={'direct','bounded'}
            pair = {r['arm']:r for r in rows}
            for field in ('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
                'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
                'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes'):
                assert pair['direct'][field]==pair['bounded'][field],(task,sample,field)
            prompt = bench.DATASET_TO_PROMPT[task].format(**item)
            if task not in bench.NO_CHAT_TASKS:
                prompt = f'[INST] {prompt} [/INST]'
            prompt_ids=tokenizer(prompt,return_tensors='pt').input_ids
            expected_tokens=min(prompt_ids.shape[1],31500)
            for arm,row in pair.items():
                assert row['task']==task and row['seed']==0
                assert row['reserved_host_tensor_bytes']==plan['expected_root_bytes'][arm]
                assert row['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
                ids = row['generated_token_ids']
                assert row['generation']['prompt_tokens']==expected_tokens
                assert row['generation']['generated_tokens']==len(ids)
                assert row['max_gen']==bench.DATASET_TO_MAXGEN[task]
                assert 0<len(ids)<=row['max_gen'] and tokenizer.eos_token_id not in ids[:-1]
                assert tokenizer.decode(ids,skip_special_tokens=True)==row['prediction']
                assert row['termination']==('eos' if ids[-1]==tokenizer.eos_token_id else 'cap')
                if row['termination']=='cap':
                    assert len(ids)==row['max_gen']
                score = 100*max(bench.DATASET_TO_METRIC[task](row['prediction'],answer,
                    all_classes=item.get('all_classes',[])) for answer in item['answers'])
                assert abs(score-row['score'])<1e-8
                assert len(row['prefill_query_shapes'])==32
                assert all(shape==[32,128,128] for shape in row['prefill_query_shapes'].values())
            pairs.append(dict(task=task,sample_id=sample,generated_tokens=len(pair['direct']['generated_token_ids']),
                              score=pair['direct']['score'],exact=True))
    assert len(pairs)==12 and len(generation['rows'])==24
    assert interleaved['hot']==1024 and interleaved['cold']==2048 and interleaved['bounded_interleaved']
    assert not interleaved['direct_interleaved'] and len(interleaved['rows'])==9
    prompt_path=ROOT/'results/remediation_v3/latency_prompt.txt'
    assert sha(prompt_path)==interleaved['prompt_sha256']
    ids=tokenizer(prompt_path.read_text(),return_tensors='pt').input_ids
    inputs=dict(A=ids[:,:4096],B=ids[:,:8192],C=ids[:,8192:12288])
    for name,frames in (('A',13),('B',25),('C',13)):
        assert hashlib.sha256(inputs[name].numpy().tobytes()).hexdigest()==interleaved['input_sha256'][name]
        rows=[r for r in interleaved['rows'] if r['request']==name]
        assert len(rows)==3 and {r['mode'] for r in rows}=={'gpu','cpu','interleaved_cpu'}
        for row in rows:
            assert len(row['predicted_token_ids'])==len(row['every_boundary_sha256'])==frames
            for field in ('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters'):
                assert row[field]==rows[0][field],(name,field)
    result=dict(verified=True,generation_pairs=12,tasks=6,seeds=[0],interleaved_requests=3,
        compared_modes=['gpu','cpu','interleaved_cpu'],boundary_frames_per_mode=51,
        unchanged_policy_classes_ast_verified=True,native_task_caps_and_eos_verified=True,
        expected_root_bytes=plan['expected_root_bytes'],pairs=pairs,
        evidence_sha256={name:sha(BASE/name) for name in ('generation.json','interleaved.json',
            'generation_manifest.json','source.tar.gz','source_manifest.json','plan.json')},
        verifier_sha256=sha(Path(__file__)),
        limitations='Reused development documents/one greedy seed; exact preservation, no independent quality superiority. Interleaving uses serialized Python/default model stream, fixed lengths/EOS ignored; no threaded/asynchronous serving, saturation, throughput, physicalSTT or default-budget host-RSS improvement claim.')
    output=BASE/'verified_report.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))


if __name__=='__main__':
    main()
