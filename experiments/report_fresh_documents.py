"""Verify all 300 arms and compute preregistered document-paired quality gates."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import longbench_eval as bench
from audit_unused_documents import document_key
from provenance import atomic_json, read_jsonl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    m=json.loads(args.manifest.read_text());r=json.loads(args.raw.read_text())
    assert r['passed'] and r['exact_cpu_gpu_actual_query_generation_state_and_trajectory']
    assert r['manifest_sha256']==sha(args.manifest) and len(r['rows'])==300
    source_manifest=args.manifest.with_name('evaluation_source_manifest_20261001.json')
    source=json.loads(source_manifest.read_text())
    bundle=args.manifest.with_name('evaluation_source_20261001.tar.gz')
    assert sha(bundle)==source['source_bundle_sha256']
    with tarfile.open(bundle,'r:gz') as archive:
        for path,expected in source['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==expected,path
        for path,expected in r['source_sha256'].items():
            assert source['source_sha256'][path]==expected,path
        runtime_files=sorted(p for p in source['source_sha256'] if p.startswith('src/') and p.endswith('.py'))+[
            f'experiments/{p}' for p in ('longbench_eval.py','longbench_metrics.py','latency_profile.py','ppl_eval.py','provenance.py')]
        digest=hashlib.sha256()
        for path in runtime_files:
            digest.update(path.encode());digest.update(archive.extractfile(path).read())
        assert digest.hexdigest()==r['runtime']['source_sha256']
    inventory_path=Path('results/baseline_fidelity/unused_document_inventory_20261001.json')
    assert sha(inventory_path)==m['inventory_sha256']
    used=set(json.loads(inventory_path.read_text())['excluded_normalized_context_keys'])
    seen=set();order=[];task_diffs={};task_stats={};sample_diffs=[];regressions=[]
    eos=json.loads((Path(m['model'])/'config.json').read_text())['eos_token_id']
    eos_ids=set(eos if isinstance(eos,list) else [eos])
    for task,meta in m['tasks'].items():
        data_path=Path(meta['slice_file']);source_path=Path(meta['source_file'])
        assert sha(data_path)==meta['slice_sha256'] and sha(source_path)==meta['source_sha256']
        data=read_jsonl(data_path);source_rows=read_jsonl(source_path)
        assert len(data)==20 and data==[source_rows[i] for i in meta['selected_zero_based_indices']]
        diffs=[];control_scores=[];candidate_scores=[]
        for item,sample,key in zip(data,meta['selected_sample_ids'],meta['selected_normalized_context_sha256']):
            assert sample==hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
            assert key==document_key(item) and key not in used|seen
            seen.add(key)
            index=len(order);order.append((task,sample))
            matching=[row for row in r['rows'] if row['task']==task and row['sample_id']==sample]
            assert len(matching)==3
            pair={row['arm']:row for row in matching}
            assert set(pair)=={'gpu_proxy','gpu_query','cpu_query'}
            for arm,row in pair.items():
                assert row['request_index']==index and row['normalized_context_sha256']==key
                prediction=row['prediction']
                if task in ('trec','triviaqa','samsum','lsht'):
                    prediction=prediction.lstrip('\n').split('\n')[0]
                score=100*max(bench.DATASET_TO_METRIC[task](prediction,answer,
                    all_classes=item.get('all_classes',[])) for answer in item['answers'])
                assert abs(score-row['score'])<1e-8
                count=len(row['generated_token_ids'])
                assert 0<count<=row['max_gen']==bench.DATASET_TO_MAXGEN[task]
                assert count==row['generation']['generated_tokens']==row['trajectory_frames']
                assert row['counters']['decode_steps']==count-1
                assert row['termination']==('eos' if row['generated_token_ids'][-1] in eos_ids else 'cap')
                assert row['termination']=='eos' or count==row['max_gen']
                assert 0<row['generation']['prompt_tokens']<=31500
                if arm=='gpu_proxy':assert row['prefill_query_shapes']=={}
                else:
                    assert set(row['prefill_query_shapes'])=={str(i) for i in range(32)}
                    assert all(shape==[32,min(128,row['generation']['prompt_tokens']),128] for shape in row['prefill_query_shapes'].values())
            for field in ('prediction','generated_token_ids','score','generation','max_gen','termination',
                          'prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes'):
                assert pair['gpu_query'][field]==pair['cpu_query'][field],field
            a,b=pair['gpu_proxy']['score'],pair['cpu_query']['score']
            diff=b-a;diffs.append(diff);control_scores.append(a);candidate_scores.append(b)
            record=dict(task=task,sample_id=sample,control_score=a,candidate_score=b,difference=diff,
                control_generated_tokens=len(pair['gpu_proxy']['generated_token_ids']),candidate_generated_tokens=len(pair['cpu_query']['generated_token_ids']))
            sample_diffs.append(record)
            if diff<0:regressions.append(record)
        task_diffs[task]=np.asarray(diffs,dtype=np.float64)
        task_stats[task]=dict(n=20,control_mean=float(np.mean(control_scores)),candidate_mean=float(np.mean(candidate_scores)),mean_difference=float(np.mean(diffs)),
            regressions=sum(d<0 for d in diffs),losses_over_10=sum(d<-10 for d in diffs))
    assert len(seen)==100 and len(task_diffs)==5
    expected=[];arms=['gpu_proxy','gpu_query','cpu_query']
    for index,(task,sample) in enumerate(order):
        offset=index%3
        expected.extend((index,task,sample,arm) for arm in arms[offset:]+arms[:offset])
    assert [(row['request_index'],row['task'],row['sample_id'],row['arm']) for row in r['rows']]==expected
    rng=np.random.default_rng(20261003)
    draws=np.mean(np.stack([rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1) for d in task_diffs.values()]),axis=0)
    ci=np.quantile(draws,[.025,.975]).tolist()
    report=dict(verified=True,raw_sha256=sha(args.raw),manifest_sha256=sha(args.manifest),source_bundle_sha256=sha(bundle),
        report_runner_sha256=sha(__file__),documents=100,tasks=5,arms=300,
        exact_cpu_gpu_actual_query_generation_state_and_trajectory=True,
        quality=dict(comparison='cpu actual-query minus gpu proxy',task_balanced_mean_difference=float(np.mean([d.mean() for d in task_diffs.values()])),
            paired_document_bootstrap_ci95=ci,bootstrap_draws=10000,bootstrap_seed=20261003,
            noninferiority_margin=-.5,noninferiority_passed=ci[0]>-.5,superiority_passed=ci[0]>0,per_task=task_stats,
            regression_count=len(regressions),losses_over_10_points=sum(row['difference']<-10 for row in regressions),regressions=regressions,sample_differences=sample_diffs),
        limitations='Conditional on 100 distinct exact/normalized documents, five tasks, one model/seed. NarrativeQA exhausted; semantic/training overlap unverified. Initial configured budgets same but physical resources/prefill work differ; final free-generation states vary across placement arms. No paired latency, matched faithful-baseline, controller/endurance or physical STT gate.')
    atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report['quality'].items() if k not in ('sample_differences','regressions')},indent=2))


if __name__=='__main__':
    main()
