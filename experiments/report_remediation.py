#!/usr/bin/env python3
"""Validate new runs and compute paired, task-balanced uncertainty intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
from provenance import atomic_json

TASKS = ('multifieldqa_en', 'qasper', 'narrativeqa', 'hotpotqa', 'triviaqa', 'gov_report')


def load_pair(directory, comparison):
    directory = Path(directory)
    files = [directory / f'{comparison}_short.json', directory / f'{comparison}_gov.json']
    docs = [json.loads(p.read_text()) for p in files]
    common = set(docs[0]['config']) & set(docs[1]['config'])
    variable = {'tasks', 'max_samples', 'dataset_sha256'}
    for key in common - variable:
        if docs[0]['config'][key] != docs[1]['config'][key]:
            raise ValueError(f'{comparison} short/GovReport config differs: {key}')
    if set(docs[0]['results']) & set(docs[1]['results']):
        raise ValueError(f'{comparison} has duplicate tasks')
    results = {**docs[0]['results'], **docs[1]['results']}
    if set(results) != set(TASKS):
        raise ValueError(f'{comparison} incomplete: {set(TASKS)-set(results)}')
    return docs, results


def compare(results, baseline, bootstrap=10000, seed=0, tasks=TASKS):
    rng = np.random.default_rng(seed)
    task_diffs, draws, scores = {}, [], {}
    for task in tasks:
        entries = results[task]
        a, b = entries['tieredkv'], entries[baseline]
        sa, sb = a['samples'], b['samples']
        if a['n'] != len(sa) or b['n'] != len(sb):
            raise ValueError(f'Incomplete sample records for {task}')
        ia, ib = [r['sample_id'] for r in sa], [r['sample_id'] for r in sb]
        if ia != ib or len(ia) != len(set(ia)):
            raise ValueError(f'Unpaired samples for {task} vs {baseline}')
        diff = np.array([x['score']-y['score'] for x,y in zip(sa,sb)])
        if len(diff)<2:
            raise ValueError(f'Need at least two paired samples for {task}')
        task_diffs[task] = {'n':len(diff), 'mean_difference':float(diff.mean()),
                            'se':float(diff.std(ddof=1)/np.sqrt(len(diff)))}
        draws.append(rng.choice(diff, size=(bootstrap,len(diff)),replace=True).mean(axis=1))
        scores[task]={'tieredkv':a['score'],baseline:b['score']}
    overall = np.mean(np.stack(draws),axis=0)
    return {'baseline':baseline,'per_task':task_diffs,'scores':scores,
            'task_balanced_mean_difference':float(np.mean([x['mean_difference'] for x in task_diffs.values()])),
            'paired_bootstrap_95pct_interval':np.quantile(overall,[.025,.975]).tolist(),
            'bootstrap_draws':bootstrap,'seed':seed}


def merged_config(docs, results):
    config = {k: v for k, v in docs[0]['config'].items()
              if k not in {'tasks', 'max_samples', 'dataset_sha256', 'methods'}}
    config.update(
        tasks=list(TASKS),
        methods=['full', 'streamingllm', 'h2o', 'snapkv', 'tieredkv'],
        samples_per_task={task: results[task]['tieredkv']['n'] for task in TASKS},
        dataset_sha256_by_task={**docs[0]['config']['dataset_sha256'],
                                **docs[1]['config']['dataset_sha256']},
    )
    return config


def validate_all_methods(results):
    methods = ('full', 'streamingllm', 'h2o', 'snapkv', 'tieredkv')
    for task in TASKS:
        if set(results[task]) != set(methods):
            raise ValueError(f'Missing method for {task}')
        expected = [row['sample_id'] for row in results[task]['tieredkv']['samples']]
        if len(expected) < 2 or len(expected) != len(set(expected)):
            raise ValueError(f'Invalid TieredKV samples for {task}')
        for method in methods:
            entry = results[task][method]
            ids = [row['sample_id'] for row in entry['samples']]
            if entry['n'] != len(ids) or ids != expected:
                raise ValueError(f'Unpaired/incomplete {method} samples for {task}')


def report_ablations(directory, baseline_doc, bootstrap):
    """Pair every five-task variant against the unchanged default run."""
    base_config = baseline_doc['config']
    base = baseline_doc['results']
    output = {
        'method': 'paired sample bootstrap within five fixed tasks; unweighted task mean',
        'uncertainty_scope': 'Conditional on selected tasks/samples/model/seed; intervals are not multiple-comparison adjusted',
        'baseline': {
            'score': float(np.mean([base[t]['tieredkv']['score'] for t in TASKS[:-1]])),
            'write_savings_pct': pooled_write_savings(base),
            'recent': base_config['recent'],
            'write_aware_lambda': base_config['write_aware_lambda'],
        },
        'variants': {},
    }
    for name in ('lambda_0.01', 'lambda_0.05', 'window_32', 'window_64',
                 'window_128', 'window_256'):
        document = json.loads((Path(directory) / (name + '.json')).read_text())
        config = document['config']
        varying = 'write_aware_lambda' if name.startswith('lambda_') else 'recent'
        for key, value in base_config.items():
            if key not in {'methods', varying} and config.get(key) != value:
                raise ValueError(f'{name} differs unexpectedly in {key}')
        if config['methods'] != ['tieredkv']:
            raise ValueError(f'{name} contains unexpected methods')
        results = document['results']
        if set(results) != set(TASKS[:-1]):
            raise ValueError(f'{name} has incomplete tasks')
        paired = {task: {'tieredkv': results[task]['tieredkv'],
                         'reference': base[task]['tieredkv']}
                  for task in TASKS[:-1]}
        comparison = compare(paired, 'reference', bootstrap, tasks=TASKS[:-1])
        comparison.update(
            score=float(np.mean([results[t]['tieredkv']['score'] for t in TASKS[:-1]])),
            write_savings_pct=pooled_write_savings(results),
            recent=config['recent'],
            write_aware_lambda=config['write_aware_lambda'],
        )
        output['variants'][name] = comparison
    atomic_json(Path(directory) / 'ablation_report.json', output)
    return output


def pooled_write_savings(results):
    paid = saved = 0
    for task in TASKS[:-1]:
        stats = results[task]['tieredkv']['cache_stats']
        paid += stats['paid_writes']
        saved += stats['writes_saved']
    return float(100 * saved / (paid + saved)) if paid + saved else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='results/remediation')
    p.add_argument('--bootstrap',type=int,default=10000)
    args=p.parse_args()
    equal_docs,equal=load_pair(args.directory,'equal_vram')
    if len(equal_docs[0]['results']) != 5 or len(equal_docs[1]['results']) != 1:
        raise ValueError('Expected five short tasks plus GovReport')
    validate_all_methods(equal)
    out={'method':'paired sample bootstrap within tasks; unweighted mean over six tasks',
         'uncertainty_scope':'Conditional on the six fixed tasks, selected samples, model, and seed; excludes task/seed/model variation',
         'score_units':'F1 for five QA tasks; ROUGE-L for GovReport, each on 0-100 scale',
         'runtime':equal_docs[0]['config']['runtime'],
         'equal_vram':compare(equal,'snapkv',args.bootstrap)}
    merged={'config':merged_config(equal_docs, equal), 'results':equal}
    atomic_json(Path(args.directory)/'equal_vram_merged.json',merged)
    for name in ('equal_attended','equal_total'):
        docs,res=load_pair(args.directory,name)
        comparable = ('model', 'budget', 'seed', 'max_ctx', 'runtime', 'data_dir',
                      'sink', 'recent', 'page_size', 'promote_top_pages',
                      'stt_live_floor', 'tiered_stt_expose',
                      'sttram_bifurcation_frac', 'tiered_kv_dtype',
                      'write_aware_lambda')
        for key in comparable:
            if docs[0]['config'][key] != equal_docs[0]['config'][key]:
                raise ValueError(f'{name} differs from equal-VRAM in {key}')
        for split in range(2):
            if docs[split]['config']['dataset_sha256'] != equal_docs[split]['config']['dataset_sha256']:
                raise ValueError(f'{name} uses different dataset files')
        for task in TASKS:
            tiered_ids = [s['sample_id'] for s in res[task]['tieredkv']['samples']]
            for baseline in ('full','streamingllm','h2o','snapkv'):
                baseline_ids = [s['sample_id'] for s in equal[task][baseline]['samples']]
                if tiered_ids != baseline_ids:
                    raise ValueError(f'Sample source mismatch: {task}')
                res[task][baseline]=equal[task][baseline]
        out[name]=compare(res,'snapkv',args.bootstrap)
        out[name]['tiered_configuration']=docs[0]['config']
        merged={'config':merged_config(docs, res), 'results':res}
        atomic_json(Path(args.directory)/f'{name}_merged.json',merged)
    target=Path(args.directory)/'paired_report.json'
    atomic_json(target,out)
    ablations = report_ablations(args.directory, equal_docs[0], args.bootstrap)
    print(json.dumps({name:out[name]['task_balanced_mean_difference']
                      for name in ('equal_vram', 'equal_attended', 'equal_total')}, indent=2))
    print(json.dumps({name: row['task_balanced_mean_difference']
                      for name, row in ablations['variants'].items()}, indent=2))


if __name__=='__main__':
    main()
