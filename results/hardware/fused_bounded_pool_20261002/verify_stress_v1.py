"""Independently verify frozen composition generation and request isolation."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
import longbench_eval as bench
from provenance import read_jsonl
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    primary = json.loads((BASE/'verified_systems.json').read_text())
    assert primary['verified'] and primary['advance_to_stress']
    frozen = json.loads((BASE/'source_stress_manifest.json').read_text())['source_sha256']
    native_path = ROOT/'results/hardware/fused_eviction_stress_20261002/verified_report.json'
    native = json.loads(native_path.read_text())
    assert native['verified'] and native['native_bit_and_stream_lifetime_checks']==15
    old = json.loads((native_path.parent/'source_manifest.json').read_text())['source_sha256']
    for name in ('experiments/fused_eviction_policy.py','experiments/fused_hot_compaction.py',
                 'experiments/native_hot_compact.cpp','experiments/native_hot_compact.cu',
                 'experiments/batched_eviction_policy.py','experiments/direct_shared_pool.py',
                 'experiments/bounded_pinned_pool.py','src/tiered_kv_cache.py'):
        assert frozen[name]==old[name], name
    results = {}
    evidence = {}
    with tarfile.open(BASE/'source_stress.tar.gz') as archive:
        for name,digest in frozen.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
        for name,hot,cold in (('default',1024,2048),('shared',991,1090)):
            directory=BASE/f'generation_{name}'
            manifest=json.loads((directory/'generation_manifest.json').read_text())
            generation=json.loads((directory/'generation.json').read_text())
            interleaved=json.loads((BASE/f'interleaved_{name}.json').read_text())
            assert manifest['hot']==hot and manifest['cold']==cold and manifest['seeds']==[0]
            assert generation['passed'] and generation['bounded_composition'] and interleaved['passed']
            assert generation['manifest_sha256']==sha(directory/'generation_manifest.json')
            assert generation['unchanged_native_proof_sha256']==sha(native_path)
            assert generation['native_checks']==[]  # Prior unchanged native proof, not rerun.
            assert generation['binary_sha256']==interleaved['binary_sha256']
            systems=json.loads((BASE/f'parity_{name}.json').read_text())
            assert generation['binary_sha256']==systems['fused_binary_sha256']
            for raw in (generation,interleaved):
                for path,digest in raw['source_sha256'].items():
                    assert digest==frozen[path],path
            for path,digest in interleaved['fused_source_sha256'].items():
                assert digest==frozen[path],path
            tokenizer=AutoTokenizer.from_pretrained(str((ROOT/manifest['model']).resolve()))
            roots=dict(direct=(cold+hot+1)*131072,fused=(cold+64)*131072)
            pairs=[]
            for task,entry in manifest['tasks'].items():
                source=ROOT/entry['source_file']
                assert sha(source)==entry['source_sha256']
                data=read_jsonl(source)
                for index,sample in zip(entry['selected_zero_based_indices'],entry['selected_sample_ids']):
                    item=data[index]
                    assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
                    rows=[r for r in generation['rows'] if r['sample_id']==sample]
                    assert len(rows)==2 and {r['arm'] for r in rows}=={'direct','fused'}
                    pair={r['arm']:r for r in rows}
                    for field in ('prediction','generated_token_ids','score','generation','max_gen','termination',
                        'prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters',
                        'prefill_query_shapes','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes',
                        'occupied_host_tensor_bytes'):
                        assert pair['direct'][field]==pair['fused'][field],(name,task,sample,field)
                    prompt=bench.DATASET_TO_PROMPT[task].format(**item)
                    if task not in bench.NO_CHAT_TASKS:prompt=f'[INST] {prompt} [/INST]'
                    prompt_tokens=min(tokenizer(prompt,return_tensors='pt').input_ids.shape[1],31500)
                    for arm,row in pair.items():
                        assert row['task']==task and row['seed']==0
                        assert row['reserved_host_tensor_bytes']==roots[arm]
                        assert row['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
                        ids=row['generated_token_ids']
                        assert row['generation']['prompt_tokens']==prompt_tokens
                        assert row['generation']['generated_tokens']==len(ids)
                        assert row['max_gen']==bench.DATASET_TO_MAXGEN[task]
                        assert 0<len(ids)<=row['max_gen'] and tokenizer.eos_token_id not in ids[:-1]
                        assert tokenizer.decode(ids,skip_special_tokens=True)==row['prediction']
                        assert row['termination']==('eos' if ids[-1]==tokenizer.eos_token_id else 'cap')
                        if row['termination']=='cap':assert len(ids)==row['max_gen']
                        score=100*max(bench.DATASET_TO_METRIC[task](row['prediction'],answer,
                            all_classes=item.get('all_classes',[])) for answer in item['answers'])
                        assert abs(score-row['score'])<1e-8
                        assert len(row['prefill_query_shapes'])==32
                        assert all(shape==[32,128,128] for shape in row['prefill_query_shapes'].values())
                    pairs.append(dict(task=task,sample_id=sample,exact=True,
                        generated_tokens=len(pair['fused']['generated_token_ids'])))
            assert len(pairs)==12 and len(generation['rows'])==24
            assert sum(r['compaction_counts'].get('fused_compactions',0) for r in generation['rows'])>0
            assert interleaved['hot']==hot and interleaved['cold']==cold
            assert interleaved['fused_bounded_interleaved']
            assert not any(interleaved[k] for k in ('fused_interleaved','bounded_interleaved','direct_interleaved'))
            assert len(interleaved['rows'])==9
            prompt_path=ROOT/'results/remediation_v3/latency_prompt.txt'
            assert sha(prompt_path)==interleaved['prompt_sha256']
            ids=tokenizer(prompt_path.read_text(),return_tensors='pt').input_ids
            inputs=dict(A=ids[:,:4096],B=ids[:,:8192],C=ids[:,8192:12288])
            for request,frames in (('A',13),('B',25),('C',13)):
                assert hashlib.sha256(inputs[request].numpy().tobytes()).hexdigest()==interleaved['input_sha256'][request]
                rows=[r for r in interleaved['rows'] if r['request']==request]
                assert len(rows)==3 and {r['mode'] for r in rows}=={'gpu','cpu','interleaved_cpu'}
                for row in rows:
                    assert len(row['predicted_token_ids'])==len(row['every_boundary_sha256'])==frames
                    for field in ('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters'):
                        assert row[field]==rows[0][field],(name,request,field)
            results[name]=dict(hot=hot,cold=cold,generation_pairs=12,tasks=6,seeds=[0],
                requested_roots=roots,interleaved_requests=3,boundary_frames_per_mode=51,pairs=pairs)
            for p in (directory/'generation_manifest.json',directory/'generation.json',BASE/f'interleaved_{name}.json'):
                evidence[str(p.relative_to(BASE))]=sha(p)
    output=BASE/'verified_stress.json'
    if output.exists():raise FileExistsError(output)
    result=dict(verified=True,geometries=results,advance_to_cli_combination=True,
        reused_unchanged_native_checks=15,native_proof_sha256=sha(native_path),
        systems_sha256=sha(BASE/'verified_systems.json'),evidence_sha256=evidence,
        source_archive_sha256=sha(BASE/'source_stress.tar.gz'),verifier_sha256=sha(Path(__file__)),
        limitations='Reused development documents, one greedy seed; preservation, not quality superiority. Serialized interleaving and fixed lengths, not concurrent serving or throughput. Requested pinned roots are not physical STT, allocator capacity or measured RSS.')
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='geometries'},indent=2))


if __name__=='__main__':main()
